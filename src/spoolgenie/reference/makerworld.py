"""MakerWorld public reference search & multimodal style/palette/geometry analyzer.

Uses public search indexing and direct user-supplied URLs/images/meshes.
Never attempts to defeat or bypass Cloudflare, CAPTCHAs, or authentication gates.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from spoolgenie.llm.client import UnifiedLLMClient, extract_json_object
from spoolgenie.schemas import ReferenceAsset

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def _clean_html(raw: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def _unwrap_ddg_url(href: str) -> str:
    if "uddg=" in href:
        qs = parse_qs(urlparse(href).query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return href


def search_makerworld(query: str, max_results: int = 6) -> list[ReferenceAsset]:
    """Search public MakerWorld model listings via ddgs (public search index, zero anti-bot bypass).

    Enriches results with public image thumbnails when available and falls back cleanly if offline.
    """
    q = query.strip()
    if not q:
        return []

    results: list[ReferenceAsset] = []
    seen: set[str] = set()

    # 1. Primary: ddgs public search index (reliable cross-platform, returns real makerworld.com/en/models URLs)
    try:
        from ddgs import DDGS

        ddgs = DDGS()
        raw_hits = list(ddgs.text(f"site:makerworld.com/en/models {q}", max_results=max_results + 4))
        if not raw_hits:
            raw_hits = list(ddgs.text(f"site:makerworld.com {q}", max_results=max_results + 4))

        # Best-effort public thumbnails for visual cards
        thumb_urls: list[str] = []
        try:
            img_hits = list(ddgs.images(f"makerworld {q} 3d print", max_results=max_results))
            thumb_urls = [str(im.get("thumbnail") or im.get("image") or "") for im in img_hits if im.get("thumbnail") or im.get("image")]
        except Exception:
            pass

        for idx, hit in enumerate(raw_hits):
            url = str(hit.get("href") or hit.get("url") or "").strip()
            if "makerworld.com" not in url.lower() or url in seen:
                continue
            seen.add(url)
            title = str(hit.get("title") or "").strip()
            title = re.sub(r"\s*[-|]\s*(?:Free 3D Print Model\s*[-|]\s*)?MakerWorld.*$", "", title, flags=re.I).strip()
            snippet = str(hit.get("body") or hit.get("snippet") or "").strip()
            thumb = thumb_urls[idx] if idx < len(thumb_urls) else None
            results.append(
                ReferenceAsset(
                    source="makerworld_search",
                    title=title or f"MakerWorld Model ({q})",
                    url=url,
                    thumbnail_url=thumb,
                    snippet=snippet,
                )
            )
            if len(results) >= max_results:
                break
    except Exception:
        pass

    # 2. Secondary fallback: DuckDuckGo HTML endpoint
    if not results:
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(f'site:makerworld.com/en/models {q}')}"
        try:
            resp = httpx.get(
                search_url,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
                timeout=8.0,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                body = resp.text
                links = re.findall(
                    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                    body,
                    flags=re.S,
                )
                snippets = re.findall(
                    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                    body,
                    flags=re.S,
                )
                for idx, (href, title_html) in enumerate(links):
                    url = _unwrap_ddg_url(html.unescape(href))
                    if "makerworld.com" not in url.lower() or url in seen:
                        continue
                    seen.add(url)
                    title = _clean_html(title_html)
                    title = re.sub(r"\s*[-|]\s*MakerWorld.*$", "", title, flags=re.I)
                    snippet = _clean_html(snippets[idx]) if idx < len(snippets) else ""
                    results.append(
                        ReferenceAsset(
                            source="makerworld_search",
                            title=title or f"MakerWorld Model ({q})",
                            url=url,
                            snippet=snippet,
                        )
                    )
                    if len(results) >= max_results:
                        break
        except Exception:
            pass

    if not results:
        browse_url = f"https://makerworld.com/en/search/models?keyword={quote_plus(q)}"
        results.append(
            ReferenceAsset(
                source="makerworld_search",
                title=f"Browse MakerWorld for '{q}'",
                url=browse_url,
                snippet=(
                    "Public search index did not return direct model cards (offline or rate-limited). "
                    "Open this MakerWorld search page, then paste a model URL or Ctrl+V a screenshot."
                ),
            )
        )
    return results


def inspect_makerworld_url(url: str) -> ReferenceAsset:
    """Extract readable slug/title metadata from a MakerWorld URL without bypassing anti-bot gates."""
    parsed = urlparse(url.strip())
    slug = parsed.path.rstrip("/").split("/")[-1]
    readable_title = re.sub(r"^\d+[-_]*", "", slug).replace("-", " ").strip().title() or "MakerWorld Reference"
    snippet = ""
    thumb = None

    try:
        r = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=6.0,
            follow_redirects=True,
        )
        if r.status_code == 200 and "Just a moment" not in r.text and "cf-challenge" not in r.text:
            m_title = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"', r.text)
            m_desc = re.search(r'<meta[^>]+property="og:description"[^>]+content="([^"]+)"', r.text)
            m_img = re.search(r'<meta[^>]+property="og:image"[^>]+content="([^"]+)"', r.text)
            if m_title:
                readable_title = html.unescape(m_title.group(1)).strip()
            if m_desc:
                snippet = html.unescape(m_desc.group(1)).strip()
            if m_img:
                thumb = html.unescape(m_img.group(1)).strip()
    except Exception:
        pass

    return ReferenceAsset(
        source="makerworld_url",
        title=readable_title,
        url=url.strip(),
        thumbnail_url=thumb,
        snippet=snippet or "Imported from MakerWorld URL (upload a screenshot for visual palette/geometry extraction).",
    )


def extract_dominant_hex_colors(image_path: str | Path, k: int = 4) -> list[str]:
    """Extract k dominant foreground HEX colors from a reference image using K-Means."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((256, 256))
    arr = np.asarray(img, dtype=np.float32).reshape(-1, 3)

    # Filter out near-pure white studio backgrounds when enough non-background pixels exist
    lum = 0.299 * arr[:, 0] + 0.587 * arr[:, 1] + 0.114 * arr[:, 2]
    sat = arr.max(axis=1) - arr.min(axis=1)
    fg = arr[~((lum > 240) & (sat < 15))]
    if len(fg) < 64:
        fg = arr

    n_clusters = min(k, max(1, len(np.unique(fg.astype(np.int32), axis=0))))
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = km.fit_predict(fg)
    counts = np.bincount(labels)
    order = np.argsort(-counts)

    hexes: list[str] = []
    for idx in order:
        r, g, b = np.clip(np.round(km.cluster_centers_[idx]), 0, 255).astype(int)
        hexes.append(f"#{r:02X}{g:02X}{b:02X}")
    return hexes


def analyze_reference_image(
    image_path: str | Path,
    llm: UnifiedLLMClient | None = None,
    title: str | None = None,
) -> ReferenceAsset:
    """Inspect a user-uploaded MakerWorld screenshot/photo with VLM + K-Means palette extraction."""
    path = Path(image_path)
    palette = extract_dominant_hex_colors(path, k=4)
    style_tags: list[str] = []
    notes = ""

    if llm and llm.available:
        vlm_prompt = (
            "Analyze this 3D-printable product reference image. Return ONLY valid JSON with keys: "
            '{"title": str, "style_keywords": [str], "estimated_part_count": int, '
            '"recommended_route": "cad" or "mesh" or "hybrid", "printability_notes": str}.'
        )
        raw = llm.analyze_image(path, vlm_prompt)
        parsed = extract_json_object(raw) if raw else None
        if parsed:
            title = title or str(parsed.get("title") or path.stem)
            style_tags = [str(s) for s in parsed.get("style_keywords", [])][:6]
            route_hint = parsed.get("recommended_route", "")
            p_notes = parsed.get("printability_notes", "")
            notes = f"Suggested route: {route_hint}. {p_notes}".strip()

    return ReferenceAsset(
        source="upload_image",
        title=title or path.stem,
        local_path=str(path.resolve()),
        extracted_style=style_tags or ["reference visual"],
        extracted_colors=palette,
        extracted_notes=notes or "Extracted dominant filament palette from reference image.",
    )


def analyze_reference_mesh(mesh_path: str | Path) -> ReferenceAsset:
    """Read bounding box and triangle stats from an imported reference STL/OBJ/GLB/3MF."""
    import trimesh

    path = Path(mesh_path)
    loaded = trimesh.load(path, force="mesh")
    extents = tuple(round(float(x), 1) for x in loaded.extents)
    return ReferenceAsset(
        source="upload_mesh",
        title=path.stem,
        local_path=str(path.resolve()),
        extracted_style=["imported 3D geometry"],
        extracted_notes=(
            f"Bounding box: {extents[0]} x {extents[1]} x {extents[2]} mm, "
            f"{len(loaded.faces):,} faces, watertight={bool(loaded.is_watertight)}."
        ),
    )
