"""Conversational 3D print designer agent: turns chat + references into a locked DesignSpec."""

from __future__ import annotations

import re
from typing import Any

from spoolgenie.config import BAMBU_PLA_SWATCHES, PRINTER_PROFILES
from spoolgenie.knowledge.character_catalog import match_character_catalog
from spoolgenie.llm.client import UnifiedLLMClient
from spoolgenie.schemas import (
    DesignSpec,
    DimensionsMM,
    EngineRoute,
    FilamentSlot,
    FunctionalOp,
    JointType,
    MulticolorMode,
    ProjectState,
)

COLOR_KEYWORDS: dict[str, tuple[str, str]] = {
    "黑": ("Matte Charcoal", "#2B2D30"),
    "哑光黑": ("Matte Charcoal", "#2B2D30"),
    "炭黑": ("Matte Charcoal", "#2B2D30"),
    "black": ("Matte Charcoal", "#2B2D30"),
    "charcoal": ("Matte Charcoal", "#2B2D30"),
    "白": ("Matte Ivory White", "#F2EFE9"),
    "象牙白": ("Matte Ivory White", "#F2EFE9"),
    "white": ("Matte Ivory White", "#F2EFE9"),
    "橙": ("Matte Mandarin Orange", "#F36C21"),
    "橘": ("Matte Mandarin Orange", "#F36C21"),
    "orange": ("Matte Mandarin Orange", "#F36C21"),
    "红": ("Matte Scarlet Red", "#C9302C"),
    "red": ("Matte Scarlet Red", "#C9302C"),
    "蓝": ("Matte Marine Blue", "#1E4C8A"),
    "冰蓝": ("Matte Ice Blue", "#8BBEE8"),
    "blue": ("Matte Marine Blue", "#1E4C8A"),
    "绿": ("Matte Grass Green", "#4D9A4E"),
    "green": ("Matte Grass Green", "#4D9A4E"),
    "黄": ("Matte Lemon Yellow", "#F4D03F"),
    "yellow": ("Matte Lemon Yellow", "#F4D03F"),
    "粉": ("Matte Sakura Pink", "#F2B6C1"),
    "pink": ("Matte Sakura Pink", "#F2B6C1"),
    "紫": ("Matte Lilac Purple", "#A28BC7"),
    "purple": ("Matte Lilac Purple", "#A28BC7"),
    "灰": ("Matte Ash Gray", "#8D9297"),
    "gray": ("Matte Ash Gray", "#8D9297"),
    "grey": ("Matte Ash Gray", "#8D9297"),
    "棕": ("Matte Latte Brown", "#9C7A54"),
    "木": ("Matte Latte Brown", "#9C7A54"),
    "brown": ("Matte Latte Brown", "#9C7A54"),
    "金": ("Silk Gold", "#D4AF37"),
    "gold": ("Silk Gold", "#D4AF37"),
    "银": ("Silk Silver", "#C0C5CE"),
    "silver": ("Silk Silver", "#C0C5CE"),
}

ORGANIC_HINTS = (
    "手办", "玩偶", "公仔", "雕像", "雕塑", "猫", "狗", "龙", "卡通", "可爱", "萌",
    "吉祥物", "头盔", "面具", "角色", "怪兽", "盆栽造型", "figurine", "statue",
    "character", "dragon", "cat", "dog", "sculpture", "cute", "toy", "mascot", "creature",
)

CAD_CATEGORIES: dict[str, tuple[str, ...]] = {
    "headphone_stand": ("耳机架", "耳机支架", "headphone", "headset stand"),
    "phone_stand": ("手机架", "手机支架", "平板支架", "phone stand", "tablet stand"),
    "desk_organizer": ("收纳盒", "笔筒", "整理盒", "桌面收纳", "organizer", "tray", "caddy", "pen holder"),
    "cable_clip": ("理线器", "理线夹", "固线", "cable clip", "cord holder", "cable organizer"),
    "enclosure": ("外壳", "盒子", "机箱", "保护壳", "enclosure", "case", "project box"),
    "coaster_sign": ("杯垫", "铭牌", "名牌", "牌子", "coaster", "nameplate", "badge", "sign"),
    "planter_vase": ("花盆", "花瓶", "笔杯", "planter", "vase", "pot"),
}


SYSTEM_PROMPT = """You are SpoolGenie, a senior industrial & 3D-printing product designer specializing in Bambu Lab AMS multicolor printing, glue-free mechanical joints, parametric build123d CAD, and 3D sculpted functional hybrids.

Understand the user's intent carefully:
A) If the user is asking for ideas/suggestions (e.g., "想有一些功能，你有啥建议", "what functional ideas work for WALL-E?", "给我几个方案"):
   - Propose 3 to 4 concrete, creative, print-friendly functional designs tailored specifically to THAT character or object's natural geometry (e.g., for WALL-E: 1. 方箱胸腔掏空笔筒/桌面收纳+眼镜架, 2. 双机械臂倾斜手机/Switch充电支架+履带理线槽, 3. 胸口嵌入式MagSafe无线充/AirPods充电座, 4. 磁吸便签/桌面小夜灯底座).
   - Pick the #1 best fit as the active preview `spec` so the user can immediately see a visual 2D render, AND clearly ask them to reply with 1/2/3/4 or tweak dimensions/colors/AMS units!
B) Choose the right `route`:
   - "cad": pure functional/mechanical parts with exact geometry (clamps, mounts, stands, organizers, enclosures, gears, brackets, holders, adapters). Set `category` to `"custom"` whenever the object is NOT one of the 7 basic built-ins (`headphone_stand`, `phone_stand`, `desk_organizer`, `cable_clip`, `enclosure`, `coaster_sign`, `planter_vase`) — SpoolGenie will synthesize custom build123d Python CAD code for ANY shape!
   - "mesh": pure character / figurine / sculpture / toy / statue / bust.
   - "hybrid": a recognizable character or organic sculpture WITH real practical utility carved directly into the 3D character body (e.g., WALL-E pen holder, Totoro phone charger stand, Iron Man MagSafe dock, Groot planter, Pikachu keychain).
C) For "hybrid" designs, ALWAYS populate `functional_ops` using one or more of these exact operations:
   - `{"op": "hollow_cavity", "params": {"shape": "round" or "square", "width_ratio": 0.55, "depth_ratio": 0.65}}` (pen cup / planter / desk caddy)
   - `{"op": "phone_slot", "params": {"slot_width_mm": 84, "slot_thickness_mm": 14}}` (angled phone/tablet dock + USB-C tunnel)
   - `{"op": "magsafe_recess", "params": {"diameter_mm": 56.4, "depth_mm": 5.8, "face": "front" or "top"}}` (MagSafe / wireless puck)
   - `{"op": "cable_comb", "params": {"slots": 3}}` (rear desktop cord organizer)
   - `{"op": "magnet_sockets", "params": {"diameter_mm": 6.2, "depth_mm": 2.4}}` (underside neodymium magnets)
   - `{"op": "plinth_base", "params": {"height_mm": 5}}` (anti-tip flat base)
   - `{"op": "keychain_loop", "params": {"outer_radius_mm": 7, "inner_radius_mm": 3.4}}`
D) CRITICAL COLOR & APPEARANCE RULES:
   - NEVER overwrite a character's iconic palette (e.g., WALL-E yellow/gray/charcoal/white, Iron Man red/gold/charcoal/cyan, Stitch blue/ice-blue/pink/black, Minion yellow/blue/silver/black) with generic black/orange unless the user explicitly asks to change colors!
   - Write `prompt_2d` in crisp descriptive English showing the EXACT character/object + its functional form as ONE centered 3D-printable matte PLA product on a PURE SOLID WHITE background, isometric 3/4 view, soft even studio lighting, chunky printable limbs, no text, no watermark.
   - Reply concisely in the user's language (Chinese if the user wrote Chinese).

Return ONLY a valid JSON object matching this exact schema:
{
  "assistant_reply": "helpful response (include 3-4 numbered creative functional ideas if user asked for suggestions!)",
  "spec": {
    "title": "short descriptive title",
    "summary": "1-sentence summary",
    "category": "custom | headphone_stand | phone_stand | desk_organizer | cable_clip | enclosure | coaster_sign | planter_vase | figurine | pixar_walle | ...",
    "route": "cad | mesh | hybrid",
    "dimensions": {
      "width_mm": 105,
      "depth_mm": 98,
      "height_mm": 120,
      "wall_thickness_mm": 3.2,
      "clearance_mm": 0.25
    },
    "filaments": [
      {"slot": 1, "name": "Matte Lemon Yellow", "hex_color": "#E8A825", "material": "PLA", "role": "main body"}
    ],
    "functional_features": ["hollow chest pen organizer", "flat treads"],
    "functional_ops": [
      {"op": "hollow_cavity", "params": {"shape": "square", "width_ratio": 0.56, "depth_ratio": 0.62}}
    ],
    "style_keywords": ["WALL-E", "functional desk buddy"],
    "prompt_2d": "Cute 3D-printable WALL-E robot desk pen organizer ...",
    "parametric_params": {}
  },
  "ready_to_confirm": true
}
"""


class ConversationalDesigner:
    """Maintains conversational design state with LLM intelligence and deterministic offline fallback."""

    def __init__(self, llm: UnifiedLLMClient) -> None:
        self.llm = llm

    def process_message(self, state: ProjectState, user_message: str) -> tuple[DesignSpec, str]:
        printer = PRINTER_PROFILES.get(state.spec.printer_id, PRINTER_PROFILES["bambu_p1s"])

        # First run fast heuristic parsing so explicit mm/cm/color overrides are never missed
        heuristic_spec = self._heuristic_update(state.spec, user_message, state)

        if self.llm.available:
            ref_context = [
                {
                    "title": r.title,
                    "style": r.extracted_style,
                    "colors": r.extracted_colors,
                    "notes": r.extracted_notes or r.snippet,
                }
                for r in state.references[-4:]
            ]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Printer: {printer.name} (bed {printer.bed_x_mm}x{printer.bed_y_mm}x{printer.bed_z_mm} mm, "
                        f"user AMS units={heuristic_spec.ams_units}, multicolor_mode={heuristic_spec.multicolor_mode.value}).\n"
                        f"Baseline parsed spec JSON (preserve character palette & route unless user explicitly changes them!):\n"
                        f"{heuristic_spec.model_dump_json(indent=2)}\n"
                        f"MakerWorld / uploaded references:\n{ref_context}\n"
                        f"User message:\n{user_message}"
                    ),
                },
            ]
            parsed = self.llm.chat_json(messages, temperature=0.25)
            if parsed and isinstance(parsed.get("spec"), dict):
                merged = self._merge_llm_spec(
                    heuristic_spec,
                    parsed["spec"],
                    printer.bed_x_mm,
                    printer.bed_y_mm,
                    printer.bed_z_mm,
                    user_message=user_message,
                )
                reply = str(parsed.get("assistant_reply") or "").strip()
                if reply:
                    return merged, reply

        # Deterministic offline reply (still works smoothly with zero LLM!)
        reply = self._build_fallback_reply(heuristic_spec, user_message)
        return heuristic_spec, reply

    def _merge_llm_spec(
        self,
        base: DesignSpec,
        raw: dict[str, Any],
        max_x: float,
        max_y: float,
        max_z: float,
        user_message: str = "",
    ) -> DesignSpec:
        data = base.model_dump()
        user_mentioned_color = any(k in user_message.lower() for k in COLOR_KEYWORDS) or bool(re.search(r"#[0-9A-Fa-f]{6}", user_message))
        is_char_base = base.category in ("figurine",) or any(
            base.category.startswith(p)
            for p in (
                "marvel_", "dc_", "disney_", "pixar_", "universal_", "dreamworks_",
                "scifi_", "starwars_", "hp_", "movie_", "anime_", "game_", "toy_",
                "animal_", "plant_", "decor_", "veh_", "food_", "hobby_", "myth_", "season_",
            )
        )

        for field in ("title", "summary", "prompt_2d"):
            if raw.get(field):
                data[field] = str(raw[field])
        # Keep character category ID intact when character preset matched so we never fall back to a generic template!
        if raw.get("category") and not is_char_base:
            data["category"] = str(raw["category"])

        route_val = str(raw.get("route", "")).lower()
        if route_val in ("cad", "mesh", "hybrid"):
            data["route"] = route_val

        if isinstance(raw.get("dimensions"), dict):
            dims = dict(data["dimensions"])
            for k, lim in (("width_mm", max_x), ("depth_mm", max_y), ("height_mm", max_z)):
                v = raw["dimensions"].get(k)
                if isinstance(v, (int, float)) and 5 <= v <= lim:
                    dims[k] = round(float(v), 1)
            for k, lo, hi in (("wall_thickness_mm", 0.8, 25.0), ("clearance_mm", 0.0, 4.0)):
                v = raw["dimensions"].get(k)
                if isinstance(v, (int, float)) and lo <= v <= hi:
                    dims[k] = round(float(v), 2)
            data["dimensions"] = dims

        # Canon-First Palette Lock: preserve curated canonical character/animal/plant colors & roles
        # unless the user explicitly asks to repaint with different colors!
        explicit_repaint = user_mentioned_color and any(
            k in user_message.lower() for k in ("改", "换", "涂成", "变成", "change", "repaint", "replace", "swap", "make it")
        )
        if isinstance(raw.get("filaments"), list) and raw["filaments"] and (not is_char_base or explicit_repaint):
            cleaned: list[dict[str, Any]] = []
            for idx, item in enumerate(raw["filaments"][:16], start=1):
                if not isinstance(item, dict):
                    continue
                hex_c = str(item.get("hex_color", "#2B2D30")).strip()
                if not re.match(r"^#[0-9A-Fa-f]{6}$", hex_c):
                    hex_c = list(BAMBU_PLA_SWATCHES.values())[(idx - 1) % len(BAMBU_PLA_SWATCHES)]
                cleaned.append(
                    {
                        "slot": idx,
                        "name": str(item.get("name") or f"AMS Color {idx}"),
                        "hex_color": hex_c.upper(),
                        "material": str(item.get("material") or "PLA"),
                        "role": str(item.get("role") or ("main body" if idx == 1 else f"accent {idx}")),
                    }
                )
            if cleaned:
                data["filaments"] = cleaned

        valid_ops = {"hollow_cavity", "phone_slot", "magsafe_recess", "magnet_sockets", "cable_comb", "plinth_base", "keychain_loop"}
        if isinstance(raw.get("functional_ops"), list):
            parsed_ops: list[dict[str, Any]] = []
            for item in raw["functional_ops"]:
                if isinstance(item, dict) and item.get("op") in valid_ops:
                    parsed_ops.append({"op": item["op"], "params": item.get("params") or {}})
            if parsed_ops:
                data["functional_ops"] = parsed_ops

        for list_field in ("functional_features", "style_keywords"):
            if isinstance(raw.get(list_field), list) and raw[list_field]:
                data[list_field] = [str(x) for x in raw[list_field][:8]]

        if isinstance(raw.get("parametric_params"), dict):
            params = dict(data.get("parametric_params", {}))
            for k, v in raw["parametric_params"].items():
                if isinstance(v, (int, float, str, bool)):
                    params[str(k)] = v
            data["parametric_params"] = params

        data["confirmed_by_user"] = False
        return DesignSpec.model_validate(data)

    def _heuristic_update(self, current: DesignSpec, text: str, state: ProjectState) -> DesignSpec:
        spec = current.model_copy(deep=True)
        low = text.lower()

        # Check curated Disney / Universal / Marvel / Anime character catalog first!
        char_preset = match_character_catalog(text)
        if char_preset is not None:
            is_new_subject = (not state.history) or (current.category != char_preset.id)
            spec.title = char_preset.name
            spec.category = char_preset.id
            spec.route = EngineRoute(char_preset.route)
            if is_new_subject:
                w, d, h = char_preset.default_dims_mm
                spec.dimensions = DimensionsMM(width_mm=w, depth_mm=d, height_mm=h)
                spec.filaments = [
                    FilamentSlot(slot=i + 1, name=cname, hex_color=chex, role=crole)
                    for i, (cname, chex, crole) in enumerate(char_preset.filaments)
                ]
            spec.prompt_2d = char_preset.prompt_2d
            spec.functional_features = [char_preset.print_tips, char_preset.no_ams_split_tip]

        # Detect category & route
        matched_cat = None
        for cat, kws in CAD_CATEGORIES.items():
            if any(k in low for k in kws):
                matched_cat = cat
                break

        is_organic = (char_preset is not None) or any(k in low for k in ORGANIC_HINTS)

        if char_preset is None:
            if matched_cat and is_organic:
                spec.category = matched_cat
                spec.route = EngineRoute.HYBRID
            elif matched_cat:
                spec.category = matched_cat
                spec.route = EngineRoute.CAD
            elif is_organic:
                spec.category = "figurine"
                spec.route = EngineRoute.MESH
            elif not state.history:
                # Arbitrary functional/mechanical object requested by user -> custom LLM build123d synthesis!
                spec.category = "custom"
                spec.route = EngineRoute.CAD
        else:
            # If user wants a character PLUS utility/function, upgrade to HYBRID with smart functional_ops!
            wants_func = any(
                k in low
                for k in (
                    "功能", "实用", "支架", "笔筒", "收纳", "充电", "magsafe", "磁吸",
                    "理线", "挂钩", "花盆", "钥匙扣", "functional", "holder", "stand",
                    "organizer", "charger", "planter", "keychain", "utility",
                )
            )
            if wants_func or char_preset.route == "hybrid":
                spec.route = EngineRoute.HYBRID
                inferred_ops: list[FunctionalOp] = []
                if any(k in low for k in ("笔筒", "收纳", "花盆", "pen", "organizer", "planter", "caddy", "储物")):
                    inferred_ops.append(FunctionalOp(op="hollow_cavity", params={"shape": "square" if "walle" in char_preset.id else "round", "width_ratio": 0.56, "depth_ratio": 0.62}))
                if any(k in low for k in ("手机", "平板", "phone", "tablet", "switch", "支架")):
                    inferred_ops.append(FunctionalOp(op="phone_slot", params={"slot_width_mm": 84.0, "slot_thickness_mm": 14.0}))
                if any(k in low for k in ("无线充", "magsafe", "airpods", "充电座", "charger")):
                    inferred_ops.append(FunctionalOp(op="magsafe_recess", params={"diameter_mm": 56.4, "depth_mm": 5.8, "face": "front"}))
                if any(k in low for k in ("理线", "cable", "线槽")):
                    inferred_ops.append(FunctionalOp(op="cable_comb", params={"slots": 3}))
                if any(k in low for k in ("磁吸", "磁铁", "magnet")):
                    inferred_ops.append(FunctionalOp(op="magnet_sockets", params={"diameter_mm": 6.2, "depth_mm": 2.4}))
                if any(k in low for k in ("钥匙扣", "挂件", "keychain")):
                    inferred_ops.append(FunctionalOp(op="keychain_loop", params={"outer_radius_mm": 7.0, "inner_radius_mm": 3.4}))
                if not inferred_ops:
                    # Default useful desktop utility for a functional character hybrid: hollow desk caddy/pen holder!
                    inferred_ops.append(FunctionalOp(op="hollow_cavity", params={"shape": "square" if "walle" in char_preset.id else "round", "width_ratio": 0.56, "depth_ratio": 0.60}))
                spec.functional_ops = inferred_ops

        # Set sensible default dimensions per category on first turn
        if not state.history:
            defaults = {
                "headphone_stand": (115.0, 105.0, 220.0, "Desktop Headphone Stand"),
                "phone_stand": (82.0, 95.0, 115.0, "Angled Phone Stand"),
                "desk_organizer": (150.0, 100.0, 85.0, "Modular Desk Organizer"),
                "cable_clip": (76.0, 28.0, 20.0, "5-Slot Desk Cable Clip"),
                "enclosure": (110.0, 75.0, 42.0, "Ventilated Project Enclosure"),
                "coaster_sign": (96.0, 96.0, 6.0, "Hex Multicolor Coaster"),
                "planter_vase": (105.0, 105.0, 130.0, "Ribbed Geometric Planter"),
                "figurine": (95.0, 85.0, 120.0, "Sculpted Desk Figurine"),
            }
            if spec.category in defaults:
                w, d, h, title = defaults[spec.category]
                spec.dimensions = DimensionsMM(width_mm=w, depth_mm=d, height_mm=h)
                spec.title = title

        # Parse explicit dimensions: e.g. "220mm", "22cm", "120x80x50 mm"
        triple = re.search(
            r"(\d+(?:\.\d+)?)\s*[xX×\*]\s*(\d+(?:\.\d+)?)\s*[xX×\*]\s*(\d+(?:\.\d+)?)\s*(mm|cm)?",
            text,
        )
        if triple:
            scale = 10.0 if (triple.group(4) or "mm").lower() == "cm" else 1.0
            spec.dimensions.width_mm = round(float(triple.group(1)) * scale, 1)
            spec.dimensions.depth_mm = round(float(triple.group(2)) * scale, 1)
            spec.dimensions.height_mm = round(float(triple.group(3)) * scale, 1)
        else:
            for attr, patterns in (
                ("height_mm", (r"(?:高|高度|height|tall)\D{0,6}(\d+(?:\.\d+)?)\s*(mm|cm)",)),
                ("width_mm", (r"(?:宽|宽度|width|wide)\D{0,6}(\d+(?:\.\d+)?)\s*(mm|cm)",)),
                ("depth_mm", (r"(?:深|深度|长|长度|depth|length)\D{0,6}(\d+(?:\.\d+)?)\s*(mm|cm)",)),
            ):
                for pat in patterns:
                    m = re.search(pat, text, flags=re.I)
                    if m:
                        scale = 10.0 if m.group(2).lower() == "cm" else 1.0
                        val = max(6.0, min(350.0, round(float(m.group(1)) * scale, 1)))
                        setattr(spec.dimensions, attr, val)

        # Detect AMS unit count & glue-free split printing preferences from chat
        if any(k in low for k in ("没有ams", "无ams", "没ams", "no ams", "without ams", "分块打印", "分色打印", "拆件")):
            if any(k in low for k in ("没有ams", "无ams", "没ams", "no ams", "without ams")):
                spec.ams_units = 0
            if any(k in low for k in ("单色", "monochrome", "single color")):
                spec.multicolor_mode = MulticolorMode.MONOCHROME
            else:
                spec.multicolor_mode = MulticolorMode.SPLIT_NO_AMS
        else:
            m_ams = re.search(r"(\d)\s*(?:台|个)?\s*ams", low)
            if m_ams:
                spec.ams_units = max(0, min(4, int(m_ams.group(1))))
                spec.multicolor_mode = MulticolorMode.SPLIT_NO_AMS if spec.ams_units == 0 else MulticolorMode.AMS

        if any(k in low for k in ("燕尾", "dovetail")):
            spec.joint_type = JointType.DOVETAIL
        elif any(k in low for k in ("卡扣", "snap", "定位销", "插销")):
            spec.joint_type = JointType.SNAP_PIN
        elif any(k in low for k in ("榫卯", "tenon", "mortise")):
            spec.joint_type = JointType.PRESS_TENON

        # Parse colors from message or explicit HEX codes
        found_colors: list[tuple[str, str]] = []
        for hx in re.findall(r"#[0-9A-Fa-f]{6}", text):
            found_colors.append((f"Custom {hx.upper()}", hx.upper()))
        for kw, pair in COLOR_KEYWORDS.items():
            if kw in low and pair[1] not in [c[1] for c in found_colors]:
                found_colors.append(pair)

        max_capture = max(4, spec.ams_units * 4) if spec.ams_units > 0 else 4
        if found_colors:
            roles = ["main body", "accent trim", "base / detail", "logo / highlight"]
            spec.filaments = [
                FilamentSlot(slot=i + 1, name=name, hex_color=hx, role=roles[i % len(roles)])
                for i, (name, hx) in enumerate(found_colors[:max_capture])
            ]
        elif state.references and state.references[-1].extracted_colors and not state.history:
            ref_cols = state.references[-1].extracted_colors[:max_capture]
            spec.filaments = [
                FilamentSlot(slot=i + 1, name=f"Ref Color {i+1}", hex_color=hx, role=("main body" if i == 0 else f"accent {i}"))
                for i, hx in enumerate(ref_cols)
            ]

        # Extract notable features
        feature_map = {
            "cable_slot": ("理线", "线槽", "cable", "cord"),
            "weighted_base": ("加重", "稳固", "底座", "base"),
            "ventilation": ("散热", "通风", "vent"),
            "compartments": ("分区", "隔层", "格", "compartment", "slot"),
            "cyber_chamfer": ("赛博", "机甲", "科技", "cyber", "futuristic", "chamfer"),
        }
        feats = list(spec.functional_features)
        for feat_name, kws in feature_map.items():
            if any(k in low for k in kws) and feat_name not in feats:
                feats.append(feat_name)
        spec.functional_features = feats

        if not spec.summary:
            spec.summary = text[:160].strip()

        color_names = ", ".join(f"{f.name}" for f in spec.filaments[:4])
        spec.prompt_2d = (
            f"Isometric 3/4 studio product render of a 3D-printable {spec.title}, "
            f"matte PLA colors: {color_names}, chunky printable geometry, "
            f"isolated single object on pure white background, soft studio light, no text, no letters"
        )
        spec.confirmed_by_user = False
        return spec

    def _build_fallback_reply(self, spec: DesignSpec, user_message: str) -> str:
        is_zh = bool(re.search(r"[\u4e00-\u9fff]", user_message))
        colors = " / ".join(f"AMS 槽{f.slot}: {f.name} ({f.hex_color})" for f in spec.filaments)
        dims = spec.dimensions
        if is_zh:
            route_cn = {
                EngineRoute.CAD: "参数化 CAD (build123d，尺寸精确到 0.05 mm)",
                EngineRoute.MESH: "AI 有机造型网格 (本地 RTX GPU / 云端 3D 引擎)",
                EngineRoute.HYBRID: "混合引擎 (参数化精确底座 + 造型装饰)",
            }[spec.route]
            return (
                f"已为你整理方案 **{spec.title}**：\n"
                f"- **生成路线**：{route_cn}\n"
                f"- **目标尺寸**：`{dims.width_mm:.0f} × {dims.depth_mm:.0f} × {dims.height_mm:.0f} mm`（壁厚 `{dims.wall_thickness_mm} mm`）\n"
                f"- **拓竹 AMS 配色**：{colors}\n\n"
                f"已同步生成 2D 效果图预览。你可以继续对话调整颜色、尺寸或细节；确认无误后点击 **「确认并生成 3D 打印文件」**。"
            )
        return (
            f"Updated design spec for **{spec.title}**:\n"
            f"- **Engine Route**: `{spec.route.value}`\n"
            f"- **Dimensions**: `{dims.width_mm:.0f} x {dims.depth_mm:.0f} x {dims.height_mm:.0f} mm`\n"
            f"- **Bambu AMS Palette**: {colors}\n\n"
            f"Review the 2D concept preview, request any tweaks, or confirm to generate the 3D model and Bambu `.3mf`."
        )
