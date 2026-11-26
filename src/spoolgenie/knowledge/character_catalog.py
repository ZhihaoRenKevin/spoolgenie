"""Modular loader for SpoolGenie's 100+ silent built-in movie, anime, game, animal, plant, and decor priors.

All entries live as clean JSON files under `src/spoolgenie/knowledge/data/` so community contributors
can add new subjects via tiny JSON PRs without touching Python code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharacterPreset:
    id: str
    franchise: str
    name: str
    name_zh: str
    keywords: tuple[str, ...]
    route: str  # "mesh", "cad", "hybrid"
    default_dims_mm: tuple[float, float, float]
    filaments: tuple[tuple[str, str, str], ...]
    prompt_2d: str
    print_tips: str
    no_ams_split_tip: str


DATA_DIR = Path(__file__).parent / "data"


def _build_compact_prompt(subject_desc: str) -> str:
    """Keep every prompt safely under 55 words (<77 SDXL CLIP tokens) so 'no text' is never truncated."""
    return (
        f"Cute 3D-printable matte PLA model of {subject_desc}, "
        "chunky sturdy proportions, flat base, crisp 4-color matte blocks, "
        "single centered object on pure white studio background, soft light, 3/4 view, no text, no letters"
    )


def _load_catalog() -> dict[str, CharacterPreset]:
    catalog: dict[str, CharacterPreset] = {}
    if not DATA_DIR.exists():
        return catalog
    for json_file in sorted(DATA_DIR.glob("*.json")):
        try:
            items = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        for raw in items:
            pid = str(raw["id"])
            dims_raw = raw.get("dims") or [96, 88, 118]
            dims = (float(dims_raw[0]), float(dims_raw[1]), float(dims_raw[2]))
            filaments = tuple(
                (str(f[0]), str(f[1]), str(f[2]))
                for f in (raw.get("filaments") or [["Matte Charcoal", "#2B2D30", "body"]])
            )
            prompt = str(raw.get("prompt_2d") or _build_compact_prompt(str(raw["desc"])))
            catalog[pid] = CharacterPreset(
                id=pid,
                franchise=str(raw.get("franchise", "General")),
                name=str(raw["name"]),
                name_zh=str(raw.get("name_zh", raw["name"])),
                keywords=tuple(str(k).lower() for k in raw.get("keywords", [])),
                route=str(raw.get("route", "mesh")),
                default_dims_mm=dims,
                filaments=filaments,
                prompt_2d=prompt,
                print_tips=str(
                    raw.get("print_tips", "0.16 mm layer height, matte PLA finish, flat bed contact.")
                ),
                no_ams_split_tip=str(
                    raw.get(
                        "no_ams_split_tip",
                        "Split base and contrasting accents with glue-free dovetail or snap-pin for 0-AMS printing.",
                    )
                ),
            )
    return catalog


CHARACTER_CATALOG: dict[str, CharacterPreset] = _load_catalog()


def match_character_catalog(text: str) -> CharacterPreset | None:
    """Silently match any user prompt against 100+ movies, anime, games, animals, plants, and classics."""
    low = text.lower()
    best_preset: CharacterPreset | None = None
    best_kw_len = 0
    for preset in CHARACTER_CATALOG.values():
        for kw in preset.keywords:
            if kw in low and len(kw) > best_kw_len:
                best_kw_len = len(kw)
                best_preset = preset
    return best_preset


def list_local_loras(lora_dir: str | Path = "loras") -> list[dict[str, str]]:
    """Discover user-placed or locally-trained .safetensors LoRA files."""
    root = Path(lora_dir)
    if not root.exists():
        return []
    return [{"name": p.stem, "path": str(p.resolve())} for p in sorted(root.rglob("*.safetensors"))]
