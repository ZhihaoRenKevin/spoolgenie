"""Printable Character Priors + MakerWorld Functional CAD Knowledge Bank (RAG + SFT seed store).

Zero-dependency lightweight lexical + tag retrieval works on any laptop (Windows/macOS/Linux, CPU-only),
and continuously grows as users ingest MakerWorld searches, local .3mf/.stl folders, or verified builds.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from spoolgenie.reference.makerworld import analyze_reference_mesh, search_makerworld


class KnowledgeEntry(BaseModel):
    id: str
    domain: Literal["character", "functional_makerworld"]
    title: str
    tags: list[str] = Field(default_factory=list)
    summary: str
    design_rules: list[str] = Field(default_factory=list)
    recommended_palette: list[str] = Field(default_factory=list)
    prompt_prefix: str = ""
    cad_snippet: str = ""
    source_url: str | None = None


# Curated, legally clean FDM-printable character/figurine style priors (no copyrighted model weights!)
CHARACTER_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "chibi_anime": {
        "label": "Chibi / Q-Version Anime Figurine (FDM-Safe)",
        "prompt_prefix": (
            "Cute chibi stylized anime collectible vinyl figurine, 2.5-head ratio, chunky FDM-printable proportions, "
            "thick sculpted hair locks with no floating thin strands, solid round display base, "
            "crisp flat matte color blocks separated cleanly for 4-color Bambu AMS printing, "
            "isolated centered single figure on pure white studio background, soft even front lighting, no text"
        ),
        "design_rules": [
            "Maintain minimum limb/hair cross-section >= 3.5 mm for 0.4 mm nozzle strength.",
            "Attach a flat 5 mm circular/hex plinth at Z=0 so figure stands without tip-over.",
            "Use bold flat matte color zones (<= 4 colors) instead of airbrush gradients to minimize AMS swaps.",
        ],
        "recommended_palette": ["#F2EFE9", "#2B2D30", "#F36C21", "#1E4C8A"],
    },
    "movie_bust": {
        "label": "Cinematic Character Bust / Statue",
        "prompt_prefix": (
            "Heroic cinematic sci-fi movie character bust sculpture mounted on beveled museum pedestal, "
            "clean sculptural folds with overhangs <= 45 degrees for support-free FDM 3D printing, "
            "matte filament clay-render finish with distinct accent armor panels, "
            "isolated on pure white background, crisp studio rim light, 3/4 view"
        ),
        "design_rules": [
            "Keep chin/collar overhangs near <= 45 deg or angle chest forward gently to minimize tree supports.",
            "Split helmet/visor or armor badge into separate AMS slot or glue-free snap-in plate.",
        ],
        "recommended_palette": ["#2B2D30", "#8D9297", "#D4AF37", "#C9302C"],
    },
    "mecha_sci_fi": {
        "label": "Mecha / Robot Articulated Collectible",
        "prompt_prefix": (
            "Stylized chamfered mecha robot desktop figurine, chunky geometric armor plates, "
            "solid sturdy stance on flat base, vivid 3-color matte PLA industrial color blocking, "
            "no thin antennas, isolated single figure on pure white studio background, isometric 3/4 view"
        ),
        "design_rules": [
            "Replace fragile thin antennas with chamfered fins >= 2.4 mm thick.",
            "Color-block chest, visor, and chassis cleanly across 3-4 AMS slots.",
        ],
        "recommended_palette": ["#F2EFE9", "#1E4C8A", "#C9302C", "#F4D03F"],
    },
    "cute_mascot": {
        "label": "Cute Animal / Cartoon Mascot Toy",
        "prompt_prefix": (
            "Adorable rounded cartoon animal mascot desk toy, smooth organic silhouette with flat seated bottom, "
            "minimal overhangs, cheerful 3-color pastel matte PLA color zones for multicolor 3D printing, "
            "pure white seamless studio background, soft diffused lighting"
        ),
        "design_rules": [
            "Flatten bottom belly/paws cleanly at Z=0 for zero-brim bed adhesion.",
            "Keep eyes, nose, and belly as crisp high-contrast color islands.",
        ],
        "recommended_palette": ["#F2EFE9", "#F2B6C1", "#2B2D30", "#F4D03F"],
    },
}


BUILTIN_KNOWLEDGE_SEED: list[KnowledgeEntry] = [
    KnowledgeEntry(
        id="mw_gridfinity_bin",
        domain="functional_makerworld",
        title="Gridfinity 42mm Modular Storage Bin Standard",
        tags=["gridfinity", "organizer", "bin", "box", "drawer", "modular", "收纳盒", "模块"],
        summary="MakerWorld's most popular modular desk/drawer organizer grid (42 mm pitch, 7 mm height unit, 0.25 mm base tolerance).",
        design_rules=[
            "Base grid pitch is 42.0 mm per unit; outer bin width = N * 42.0 - 0.5 mm clearance.",
            "Wall thickness >= 2.0 mm (5 perimeters at 0.4 mm nozzle) with 45 deg inner bottom scoop for easy part pickup.",
            "Optional 6.2 x 2.2 mm magnet sockets on underside corners (0.2 mm press-fit clearance).",
        ],
        recommended_palette=["#2B2D30", "#F36C21", "#F2EFE9"],
        cad_snippet=(
            "# Gridfinity-inspired outer clearance: W = units_x * 42.0 - 0.5, D = units_y * 42.0 - 0.5\n"
            "# Use 45-deg bottom chamfer on inner floor so parts slide out easily without supports."
        ),
    ),
    KnowledgeEntry(
        id="mw_headphone_cantilever",
        domain="functional_makerworld",
        title="MakerWorld Weighted Desk Headphone Stand with Cable Channel",
        tags=["headphone", "headset", "stand", "desk", "cable", "耳机架", "耳机支架", "理线"],
        summary="Anti-tip desktop headphone stand placing spine slightly rear of center with curved headband saddle and rear USB/audio cable pass-through.",
        design_rules=[
            "Place spine +15% toward rear of base so heavy over-ear headphones stay inside support polygon.",
            "Top headband saddle width 55-85 mm with raised front/back lips (>= 5 mm) to prevent slipping.",
            "Print base accent trim or nameplate as a contrasting slot (or glue-free dovetail slide-in for No-AMS).",
        ],
        recommended_palette=["#2B2D30", "#F36C21"],
        cad_snippet="# Rearward spine offset col_y = DEPTH * 0.16 prevents tipping; add rear cable cut in base.",
    ),
    KnowledgeEntry(
        id="mw_skadis_hook",
        domain="functional_makerworld",
        title="IKEA Skadis Pegboard Hook & Holder Standard",
        tags=["skadis", "pegboard", "hook", "wall", "mount", "洞洞板", "挂钩"],
        summary="Standard Skadis pegboard spacing (40 mm horizontal pitch, 20 mm staggered vertical rows, 5 x 15 mm pill slots, 5 mm board thickness).",
        design_rules=[
            "Hook neck thickness 4.6 mm (fits 5.0 mm slot with 0.2 mm per-side clearance), hook depth 5.2 mm for board grip.",
            "Print hooks sideways on their flank so layer lines run along the load path (3x stronger shear resistance!).",
        ],
        recommended_palette=["#F2EFE9", "#2B2D30"],
    ),
    KnowledgeEntry(
        id="mw_glue_free_joints",
        domain="functional_makerworld",
        title="Glue-Free Mechanical Joinery (Dovetail, Compliant Snap-Pin, Press Tenon)",
        tags=["joint", "dovetail", "snap", "clip", "no glue", "split", "modular", "燕尾槽", "卡扣", "免胶水", "分块"],
        summary="MakerWorld print-in-parts engineering clearances for assembling multicolor or oversize prints without superglue.",
        design_rules=[
            "Sliding dovetail clearance: 0.20-0.25 mm per face for Bambu PLA on 0.4 mm nozzle.",
            "Compliant slotted snap-pin: split peg down center with 1.0 mm relief gap + 0.16 mm retention bead + chamfered female funnel.",
            "Add dual alignment pins or rectangular tenon to eliminate single-pin axial rotation.",
        ],
        recommended_palette=["#2B2D30", "#8D9297"],
    ),
    KnowledgeEntry(
        id="mw_enclosure_m3",
        domain="functional_makerworld",
        title="Electronics / Raspberry Pi / ESP32 Ventilated Enclosure",
        tags=["enclosure", "case", "box", "electronics", "pcb", "vent", "m3", "外壳", "盒子", "散热"],
        summary="Snap-fit ventilated project enclosure with M3 standoffs, cooling louver slots, and contrasting lid bezel.",
        design_rules=[
            "M3 self-tapping pilot bore = 2.85 mm; M3 brass heat-set insert bore = 4.6 mm; M3 clearance hole = 3.4 mm.",
            "Keep vent slots <= 4.0 mm wide with 45 deg top bridges so zero supports are needed.",
            "Snap-fit lid rim clearance = 0.25-0.30 mm around inner cavity lip.",
        ],
        recommended_palette=["#2B2D30", "#1E4C8A"],
    ),
    KnowledgeEntry(
        id="mw_phone_dock",
        domain="functional_makerworld",
        title="Ergonomic 60-Degree Desk Phone / Tablet Stand",
        tags=["phone", "tablet", "stand", "dock", "charging", "手机支架", "平板支架"],
        summary="Desktop phone dock tilted ~60-65 deg with centered 18 mm USB-C cable slot and 10 mm retention lip.",
        design_rules=[
            "Backplate tilt angle 25-30 deg from vertical (60-65 deg viewing angle).",
            "Bottom cable cutout >= 18 mm wide x 25 mm tall so thick braided USB-C connectors clear cleanly.",
        ],
        recommended_palette=["#2B2D30", "#F36C21"],
    ),
    KnowledgeEntry(
        id="mw_hsw_multiboard",
        domain="functional_makerworld",
        title="Honeycomb Storage Wall (HSW) & Multiboard Hex Standard",
        tags=["hsw", "honeycomb", "multiboard", "wall", "hex", "蜂窝墙", "墙面收纳"],
        summary="MakerWorld's Honeycomb Storage Wall uses 20.0 mm flat-to-flat outer hex inserts with 0.20 mm press-fit clearance.",
        design_rules=[
            "HSW hex insert flat-to-flat = 19.8 mm (fits 20.0 mm hex pocket with 0.2 mm clearance), depth 10 mm.",
            "Chamfer leading edges 0.8 mm at 45 deg for tool-free push-in insertion.",
        ],
        recommended_palette=["#2B2D30", "#F2EFE9"],
    ),
    KnowledgeEntry(
        id="mw_magsafe_watch",
        domain="functional_makerworld",
        title="MagSafe & Apple Watch / AirPods Charging Dock Dimensions",
        tags=["magsafe", "apple watch", "airpods", "wireless", "charger", "无线充", "充电座"],
        summary="Exact press-fit puck pockets for Apple MagSafe (56.4 x 5.8 mm) and Apple Watch charger (27.8 x 6.0 mm).",
        design_rules=[
            "MagSafe puck pocket: diameter 56.4 mm, depth 5.8 mm + 6.0 mm rear cord slot + finger-push eject hole (12 mm) on back.",
            "Apple Watch puck pocket: diameter 27.8 mm, depth 6.0 mm + 5.0 mm rear wire exit groove.",
        ],
        recommended_palette=["#2B2D30", "#F2EFE9", "#D4AF37"],
    ),
    KnowledgeEntry(
        id="mw_gopro_camera",
        domain="functional_makerworld",
        title="GoPro 3-Prong Action Mount & Arca-Swiss Tripod Plate",
        tags=["gopro", "camera", "tripod", "arca", "mount", "相机", "三脚架", "云台"],
        summary="Standard GoPro 3-prong fingers (3.0 mm fin, 3.2 mm gap, 5.2 mm M5 bolt bore) and Arca-Swiss 38 mm 45-deg dovetail.",
        design_rules=[
            "GoPro prong thickness 3.0 mm, spacing 3.2 mm, through-hole 5.2 mm for M5 thumbscrew; print prongs flat on XY for max strength.",
            "Arca-Swiss quick-release plate width 38.0 mm with 45-deg beveled lower rails.",
        ],
        recommended_palette=["#2B2D30", "#F36C21"],
    ),
    KnowledgeEntry(
        id="mw_switch_sd_storage",
        domain="functional_makerworld",
        title="Nintendo Switch Cartridge, SD & microSD Storage Tolerances",
        tags=["switch", "cartridge", "sd", "microsd", "card", "卡带", "内存卡", "存储卡"],
        summary="Exact MakerWorld slot clearances for Switch game cards (21.6 x 3.8 mm), SD cards (24.6 x 2.6 mm), and microSD (11.5 x 1.5 mm).",
        design_rules=[
            "Nintendo Switch game card slot: 22.0 mm wide x 4.2 mm thick x 18 mm deep (leaves top grip exposed for finger removal).",
            "Full SD slot: 24.8 x 2.8 mm; microSD slot: 11.8 x 1.6 mm.",
        ],
        recommended_palette=["#2B2D30", "#C9302C"],
    ),
    KnowledgeEntry(
        id="mw_desk_clamp",
        domain="functional_makerworld",
        title="Under-Desk C-Clamp & Headphone / Gamepad Hanger",
        tags=["clamp", "desk", "hook", "hanger", "gamepad", "controller", "桌边夹", "挂钩", "手柄"],
        summary="Heavy-duty C-clamp printed sideways on its flank so layer lines wrap continuously around the jaw corners.",
        design_rules=[
            "Print C-clamps on their side so bending stress runs along continuous perimeter extrusions (prevents Z-layer delamination).",
            "Add +0.5 mm clearance to target desk thickness + optional M8/M10 thumbscrew bore (10.4 mm) for multi-desk clamping.",
        ],
        recommended_palette=["#2B2D30", "#1E4C8A"],
    ),
    KnowledgeEntry(
        id="mw_battery_dispenser",
        domain="functional_makerworld",
        title="AA / AAA / 18650 Battery Organizer & Gravity Dispenser",
        tags=["battery", "aa", "aaa", "18650", "dispenser", "电池", "收纳"],
        summary="Cylindrical cell clearances: AA (14.8 x 51.0 mm), AAA (11.0 x 45.0 mm), 18650 (18.8 x 66.0 mm).",
        design_rules=[
            "Leave +0.6 mm diametral clearance for gravity-fed battery chutes so wrapped cells never jam.",
            "Slope gravity ramp at 12-15 deg with bottom lip stop 6 mm tall.",
        ],
        recommended_palette=["#2B2D30", "#F4D03F"],
    ),
    KnowledgeEntry(
        id="mw_pip_hinge_box",
        domain="functional_makerworld",
        title="Print-in-Place Hinge & Rugged Latch Box Standards",
        tags=["hinge", "latch", "rugged", "box", "print in place", "铰链", "卡扣盒", "一体打印"],
        summary="MakerWorld print-in-place knuckle hinge (0.35 mm radial clearance) and snap-latch lid design rules.",
        design_rules=[
            "Print-in-place hinge pin clearance: 0.35 mm radial, 0.40 mm axial between knuckles (prevents fusing on 0.4 mm nozzle).",
            "Snap-fit latch cantilever thickness 1.6-2.0 mm (4-5 walls), undercut barb 0.8 mm, 0.25 mm mate clearance.",
        ],
        recommended_palette=["#2B2D30", "#F36C21"],
    ),
    KnowledgeEntry(
        id="mw_laptop_vertical",
        domain="functional_makerworld",
        title="Vertical Laptop / MacBook / Steam Deck Desk Stand",
        tags=["laptop", "macbook", "steam deck", "vertical", "stand", "笔记本支架", "立式支架"],
        summary="Heavy anti-tip vertical cradle with TPU/felt-compatible slot clearance (+1.5 mm) and wide base footprint.",
        design_rules=[
            "MacBook Pro 14/16 slot width 18-20 mm; Steam Deck slot 36 mm; base width >= 2.2x slot thickness to prevent tip-over.",
            "Add 45-deg lead-in chamfers at top of cradle walls so laptops slide in smoothly without scratching.",
        ],
        recommended_palette=["#2B2D30", "#8D9297"],
    ),
    KnowledgeEntry(
        id="mw_plant_pot_drainage",
        domain="functional_makerworld",
        title="Self-Watering Planter Pot & Hidden Drip Saucer",
        tags=["planter", "pot", "drainage", "saucer", "succulent", "花盆", "沥水", "托盘"],
        summary="Two-piece planter with 6 mm drainage holes, raised inner ribs, and twist-lock or snap-fit drip saucer.",
        design_rules=[
            "Use 4 perimeters + 5 bottom layers for watertight PLA/PETG drip saucers.",
            "Add 3-5 bottom drainage holes (6 mm diameter) + 3 mm standoff feet inside saucer so roots don't rot.",
        ],
        recommended_palette=["#F2EFE9", "#4D9A4E", "#9C7A54"],
    ),
    KnowledgeEntry(
        id="mw_keycap_switch",
        domain="functional_makerworld",
        title="Cherry MX Mechanical Keyboard Keycap & Switch Tester",
        tags=["keyboard", "keycap", "mx", "switch", "键盘", "键帽", "试轴器"],
        summary="Standard Cherry MX switch plate cutout is 14.0 x 14.0 mm on a 19.05 mm pitch, plate thickness 1.5 mm.",
        design_rules=[
            "MX switch snap-in cutout: 14.05 x 14.05 mm square in a 1.5 mm thick top plate; center-to-center pitch 19.05 mm.",
            "MX cross stem socket: 4.15 mm span x 1.32 mm cross slot width.",
        ],
        recommended_palette=["#2B2D30", "#F36C21", "#F2EFE9"],
    ),
    KnowledgeEntry(
        id="mw_din_vesa_mounts",
        domain="functional_makerworld",
        title="35mm DIN Rail Clip & VESA 75/100mm Mini-PC Mount",
        tags=["din", "vesa", "mini pc", "router", "mount", "导轨", "背挂", "支架"],
        summary="Industrial 35 mm DIN rail snap clip and VESA 75x75 / 100x100 mm M4 mounting patterns.",
        design_rules=[
            "VESA 75/100 mm uses M4 screws (4.4 mm clearance holes, 8.5 mm counterbore).",
            "35 mm DIN rail clip needs a 2.0 mm thick compliant spring tab on the bottom catch for snap-on installation.",
        ],
        recommended_palette=["#2B2D30", "#8D9297"],
    ),
    KnowledgeEntry(
        id="char_chibi_prior",
        domain="character",
        title="Chibi / Q-Version Anime Collectible Figurine Prior",
        tags=["anime", "chibi", "cartoon", "figurine", "character", "动漫", "卡通", "Q版", "手办", "公仔"],
        summary="FDM-optimized Q-version character sculpting rules: chunky silhouette, solid plinth, crisp 4-color AMS color blocking.",
        design_rules=CHARACTER_STYLE_PRESETS["chibi_anime"]["design_rules"],
        recommended_palette=CHARACTER_STYLE_PRESETS["chibi_anime"]["recommended_palette"],
        prompt_prefix=CHARACTER_STYLE_PRESETS["chibi_anime"]["prompt_prefix"],
    ),
    KnowledgeEntry(
        id="char_movie_bust_prior",
        domain="character",
        title="Movie / Sci-Fi Hero Bust & Mecha Figurine Prior",
        tags=["movie", "film", "hero", "bust", "statue", "mecha", "sci-fi", "电影", "角色", "机甲", "胸像", "雕像"],
        summary="Support-minimized cinematic character bust & mecha sculpture guidelines for clean matte PLA / Silk Gold prints.",
        design_rules=CHARACTER_STYLE_PRESETS["movie_bust"]["design_rules"],
        recommended_palette=CHARACTER_STYLE_PRESETS["movie_bust"]["recommended_palette"],
        prompt_prefix=CHARACTER_STYLE_PRESETS["movie_bust"]["prompt_prefix"],
    ),
]


def _tokenize(text: str) -> set[str]:
    low = text.lower()
    words = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,2}", low))
    return words


class PrintKnowledgeBase:
    """Local persistent RAG store combining built-in priors + user-ingested MakerWorld / character records."""

    def __init__(self, store_path: str | Path = "outputs/knowledge_store.jsonl") -> None:
        self.store_path = Path(store_path)
        self.entries: list[KnowledgeEntry] = list(BUILTIN_KNOWLEDGE_SEED)
        self._load_user_entries()

    def _load_user_entries(self) -> None:
        if not self.store_path.exists():
            return
        seen = {e.id for e in self.entries}
        for line in self.store_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = KnowledgeEntry.model_validate_json(line)
                if entry.id not in seen:
                    self.entries.append(entry)
                    seen.add(entry.id)
            except Exception:
                continue

    def add_entry(self, entry: KnowledgeEntry) -> None:
        existing_ids = {e.id for e in self.entries}
        if entry.id in existing_ids:
            self.entries = [entry if e.id == entry.id else e for e in self.entries]
        else:
            self.entries.append(entry)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def retrieve(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 3,
    ) -> list[KnowledgeEntry]:
        """Score entries via weighted lexical + tag match (zero external dependencies, fast on CPU)."""
        q_tokens = _tokenize(query)
        if not q_tokens:
            pool = [e for e in self.entries if domain is None or e.domain == domain]
            return pool[:top_k]

        scored: list[tuple[float, KnowledgeEntry]] = []
        for entry in self.entries:
            if domain and entry.domain != domain:
                continue
            doc_text = " ".join([entry.title, entry.summary, *entry.tags, *entry.design_rules])
            doc_tokens = _tokenize(doc_text)
            tag_tokens = _tokenize(" ".join(entry.tags))
            overlap = len(q_tokens & doc_tokens)
            tag_hits = len(q_tokens & tag_tokens)
            score = (overlap + 2.5 * tag_hits) / math.sqrt(max(len(doc_tokens), 1))
            if score >= 0.55 and (tag_hits >= 1 or overlap >= 2):
                scored.append((score, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def ingest_makerworld_topic(self, query: str, max_results: int = 6) -> list[KnowledgeEntry]:
        """Fetch public MakerWorld search entries for a functional topic and store them as retrievable design priors."""
        refs = search_makerworld(query, max_results=max_results)
        added: list[KnowledgeEntry] = []
        for idx, ref in enumerate(refs):
            slug = re.sub(r"[^a-z0-9]+", "_", ref.title.lower()).strip("_")[:36] or f"item_{idx}"
            entry = KnowledgeEntry(
                id=f"mw_live_{slug}",
                domain="functional_makerworld",
                title=ref.title,
                tags=list(_tokenize(f"{query} {ref.title}"))[:10],
                summary=ref.snippet or f"MakerWorld functional reference for {query}.",
                design_rules=[
                    f"Reference inspired by MakerWorld listing: {ref.title}",
                    "Ensure flat Z=0 bed contact, >= 2.0 mm functional walls, and 0.22 mm mechanical tolerances.",
                ],
                source_url=ref.url,
            )
            self.add_entry(entry)
            added.append(entry)
        return added

    def ingest_local_models_dir(self, folder: str | Path, domain: Literal["character", "functional_makerworld"] = "functional_makerworld") -> list[KnowledgeEntry]:
        """Scan a user's local folder of downloaded MakerWorld .3mf/.stl files and record bounding-box & structural priors."""
        root = Path(folder)
        added: list[KnowledgeEntry] = []
        if not root.exists():
            return added
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in (".stl", ".3mf", ".obj", ".glb"):
                continue
            try:
                ref = analyze_reference_mesh(p)
                slug = re.sub(r"[^a-z0-9]+", "_", p.stem.lower()).strip("_")[:40]
                entry = KnowledgeEntry(
                    id=f"local_{slug}",
                    domain=domain,
                    title=p.stem.replace("_", " ").replace("-", " ").title(),
                    tags=list(_tokenize(p.stem))[:8] + ["local_library"],
                    summary=ref.extracted_notes,
                    design_rules=[f"Observed geometry dimensions from local reference `{p.name}`: {ref.extracted_notes}"],
                )
                self.add_entry(entry)
                added.append(entry)
            except Exception:
                continue
        return added

    def match_character_preset(self, text: str) -> tuple[str, dict[str, Any]] | None:
        low = text.lower()
        if any(k in low for k in ("q版", "黏土人", "chibi", "二次元", "动漫", "anime", "卡通")):
            return "chibi_anime", CHARACTER_STYLE_PRESETS["chibi_anime"]
        if any(k in low for k in ("机甲", "高达", "机器人", "mecha", "robot", "cyber")):
            return "mecha_sci_fi", CHARACTER_STYLE_PRESETS["mecha_sci_fi"]
        if any(k in low for k in ("电影", "胸像", "雕像", "英雄", "movie", "film", "bust", "statue", "hero")):
            return "movie_bust", CHARACTER_STYLE_PRESETS["movie_bust"]
        if any(k in low for k in ("猫", "狗", "萌", "宠物", "吉祥物", "龙", "cat", "dog", "cute", "mascot")):
            return "cute_mascot", CHARACTER_STYLE_PRESETS["cute_mascot"]
        return None
