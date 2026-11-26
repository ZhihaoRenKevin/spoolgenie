# Contributing to SpoolGenie

Contributions of all kinds are welcome — new silent subject priors, MakerWorld engineering CAD tolerances, slicer integrations, UI translations, and bug fixes.

## 1. Add a New Silent Subject Prior (Zero Python Required!)

All built-in character, animal, plant, vehicle, food, and decor priors live as clean JSON arrays under `src/spoolgenie/knowledge/data/`:

- `movies_heroes.json` — Movies, superheroes, Disney/Pixar, Universal/Sci-Fi
- `anime_games.json` — Anime, manga, video games, designer toys
- `animals.json` — Pets, wildlife, marine life, cute creatures
- `plants_decor.json` — Plants, flowers, botanical trays, architecture, space decor
- `vehicles_culture_food.json` — Vehicles, food/cafe, music/gaming, mythic/cultural, seasonal gifts

Add a single entry following this schema:

```json
{
  "id": "animal_otter",
  "franchise": "Animals",
  "name": "Cute Sea Otter Holding Clam",
  "name_zh": "动物 · 抱贝壳海獭",
  "keywords": ["otter", "sea otter", "海獭", "水獭"],
  "route": "mesh",
  "dims": [92, 86, 108],
  "filaments": [
    ["Matte Latte Brown", "#8C6239", "fur body"],
    ["Matte Ivory White", "#F5F0E4", "belly & muzzle"],
    ["Matte Sakura Pink", "#F098B4", "shell"],
    ["Basic Black", "#1C1D20", "eyes & nose"]
  ],
  "desc": "adorable chubby floating sea otter clutching a pastel shell on its belly with flat stable back"
}
```

**Important Rule**: Keep `desc` concise (~15–25 words) so the compiled `prompt_2d` stays strictly `<= 55` words (< 77 SDXL CLIP tokens). Run `pytest -q` to verify automatically.

## 2. Add a MakerWorld Functional CAD Engineering Standard

Add a `KnowledgeEntry` to `BUILTIN_ENTRIES` in `src/spoolgenie/knowledge/kb.py` with exact millimeter clearances, printable overhang angles, and AMS/no-AMS splitting tips.

## 3. Local Development & Testing

```bash
uv pip install -e ".[dev]"
ruff check src tests
pytest -q
```
