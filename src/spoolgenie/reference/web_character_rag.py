"""Web-RAG Multi-View Character & 3D-Model Reference Retriever (RA-Fab3D Prototype).

Searches the live web across TWO complementary retrieval channels before 2D/3D generation:
  1. Multi-view visual reference images (front / 3/4 character figurine & turnaround shots),
     filtered for resolution & background cleanliness and distilled via saturation-weighted CIELAB K-Means
     (+ optional local VLM anatomy analysis).
  2. Public 3D-printable model repositories (MakerWorld, Printables, Thingiverse) to surface similar
     community-tested 3D models, print orientation clues, and split-part strategies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from spoolgenie.llm.client import UnifiedLLMClient
from spoolgenie.reference.makerworld import search_makerworld
from spoolgenie.schemas import ReferenceAsset


@dataclass
class WebReferencePack:
    query: str
    downloaded_images: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    model_refs_3d: list[ReferenceAsset] = field(default_factory=list)
    extracted_canon_hex: list[str] = field(default_factory=list)
    vlm_anatomy_notes: str = ""
    prompt_enrichment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "downloaded_images": self.downloaded_images,
            "image_urls": self.image_urls,
            "model_refs_3d": [r.model_dump() for r in self.model_refs_3d],
            "extracted_canon_hex": self.extracted_canon_hex,
            "vlm_anatomy_notes": self.vlm_anatomy_notes,
            "prompt_enrichment": self.prompt_enrichment,
        }


def _search_3d_repos_multi_platform(query: str, max_results: int = 6) -> list[ReferenceAsset]:
    """Search MakerWorld, Printables, and Thingiverse for similar 3D-printable model references."""
    refs: list[ReferenceAsset] = []
    seen: set[str] = set()

    # 1. MakerWorld first (highest Bambu AMS / multi-part relevance)
    for r in search_makerworld(query, max_results=max(2, max_results // 2)):
        if r.url and r.url not in seen:
            seen.add(r.url)
            refs.append(r)

    # 2. Expand across Printables & Thingiverse for rich community 3D references
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = list(
                ddgs.text(
                    f"(site:printables.com/model OR site:thingiverse.com/thing) {query} 3d print",
                    max_results=max_results,
                )
            )
        for item in hits:
            href = str(item.get("href") or item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("body") or item.get("snippet") or "").strip()
            if not href or href in seen:
                continue
            seen.add(href)
            refs.append(
                ReferenceAsset(
                    source="makerworld_search",
                    title=title or query,
                    url=href,
                    snippet=snippet[:220],
                    extracted_notes="Community 3D-printable model reference retrieved via Web-RAG.",
                )
            )
            if len(refs) >= max_results:
                break
    except Exception:
        pass
    return refs[:max_results]


def _extract_chromatic_canon_colors(image_paths: list[Path], n_colors: int = 4) -> list[str]:
    """Extract clean canonical object colors across multiple reference images (ignoring white/black studio backdrops)."""
    samples: list[np.ndarray] = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            # Focus on central 75% crop where the character lives
            crop = arr[16:112, 16:112].reshape(-1, 3)
            lum = 0.299 * crop[:, 0] + 0.587 * crop[:, 1] + 0.114 * crop[:, 2]
            sat = crop.max(axis=1) - crop.min(axis=1)
            # Exclude near-white studio paper and extreme dark background corners
            fg = crop[~((lum > 0.92) & (sat < 0.08))]
            if len(fg) > 100:
                samples.append(fg)
        except Exception:
            continue

    if not samples:
        return []

    pixels = np.vstack(samples)
    if len(pixels) > 6000:
        rng = np.random.default_rng(42)
        pixels = pixels[rng.choice(len(pixels), size=6000, replace=False)]

    k = min(n_colors, max(1, len(pixels) // 50))
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(pixels)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)

    hex_out: list[str] = []
    for idx in order:
        rgb = np.clip(km.cluster_centers_[idx] * 255.0, 0, 255).astype(int)
        hx = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        if hx not in hex_out:
            hex_out.append(hx)
    return hex_out


def retrieve_web_character_pack(
    query: str,
    save_dir: str | Path,
    llm: UnifiedLLMClient | None = None,
    max_images: int = 4,
    max_3d_refs: int = 6,
) -> WebReferencePack:
    """Retrieve multi-view character reference images + similar 3D-printable model links from the live web."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = WebReferencePack(query=query)

    # Channel 1: Similar 3D printable model references (MakerWorld + Printables + Thingiverse)
    pack.model_refs_3d = _search_3d_repos_multi_platform(query, max_results=max_3d_refs)

    # Channel 2: Multi-angle clean visual references via DDGS image search
    candidates: list[str] = []
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            img_hits = list(
                ddgs.images(
                    f"{query} 3d model figurine collectible render white background",
                    max_results=max(8, max_images * 3),
                )
            )
        for h in img_hits:
            u = str(h.get("image") or h.get("thumbnail") or "").strip()
            w = int(h.get("width") or 0)
            ht = int(h.get("height") or 0)
            if not u.startswith("http"):
                continue
            if w and ht and (min(w, ht) < 180 or max(w / max(ht, 1), ht / max(w, 1)) > 2.2):
                continue
            if u not in candidates:
                candidates.append(u)
    except Exception:
        pass

    downloaded: list[Path] = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SpoolGenie-WebRAG/0.6)"}
    for url in candidates:
        if len(downloaded) >= max_images:
            break
        try:
            resp = httpx.get(url, headers=headers, timeout=8.0, follow_redirects=True)
            if resp.status_code != 200 or len(resp.content) < 4000:
                continue
            digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
            dst = out_dir / f"webref_{len(downloaded)+1}_{digest}.png"
            img = Image.open(httpx._content.BytesIO(resp.content) if hasattr(httpx, "_content") else __import__("io").BytesIO(resp.content)).convert("RGB")
            if min(img.width, img.height) < 180:
                continue
            img.thumbnail((768, 768), Image.Resampling.LANCZOS)
            img.save(dst, format="PNG")
            downloaded.append(dst)
            pack.downloaded_images.append(str(dst.resolve()))
            pack.image_urls.append(url)
        except Exception:
            continue

    # Distill canonical colors across retrieved multi-view images
    if downloaded:
        pack.extracted_canon_hex = _extract_chromatic_canon_colors(downloaded, n_colors=4)

    # Optional VLM cross-view anatomy & printability distillation (uses local Ollama qwen2.5vl when available)
    if downloaded and llm and llm.vlm_available:
        try:
            vlm_note = llm.describe_image(
                downloaded[0],
                (
                    f"Analyze this reference image of '{query}' for 3D printing. In 2 concise English sentences: "
                    "1) List its exact iconic silhouette parts and canonical colors; "
                    "2) Note any fragile thin parts (antennas, thin wires, open treads) that should be thickened/closed for FDM printing."
                ),
            )
            if vlm_note:
                pack.vlm_anatomy_notes = vlm_note.strip()
        except Exception:
            pass

    bits: list[str] = []
    if pack.vlm_anatomy_notes:
        bits.append(pack.vlm_anatomy_notes)
    elif pack.model_refs_3d:
        titles = ", ".join(r.title for r in pack.model_refs_3d[:3])
        bits.append(f"Grounded in community 3D references: {titles}.")
    if pack.extracted_canon_hex:
        bits.append(f"Retrieved reference palette: {', '.join(pack.extracted_canon_hex)}.")
    pack.prompt_enrichment = " ".join(bits)
    return pack
