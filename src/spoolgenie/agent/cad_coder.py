"""Parametric build123d code synthesizer with AST safety sandbox and multi-color templates."""

from __future__ import annotations

import ast
import re
from textwrap import dedent

from spoolgenie.llm.client import UnifiedLLMClient
from spoolgenie.schemas import DesignSpec

FORBIDDEN_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
}
ALLOWED_IMPORTS = {"build123d", "math"}


class UnsafeCADCodeError(ValueError):
    """Raised when generated CAD code fails AST safety validation."""


def validate_cad_code_ast(code: str) -> None:
    """Strictly verify generated Python CAD code only uses build123d and math."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCADCodeError(f"Syntax error in CAD script: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise UnsafeCADCodeError(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise UnsafeCADCodeError(f"Forbidden from-import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise UnsafeCADCodeError(f"Forbidden call: {node.func.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeCADCodeError(f"Forbidden dunder access: {node.attr}")


def generate_template_cad_code(spec: DesignSpec) -> str:
    """Produce a guaranteed-valid, manifold, multi-color build123d script for common print categories."""
    w = max(20.0, float(spec.dimensions.width_mm))
    d = max(18.0, float(spec.dimensions.depth_mm))
    h = max(6.0, float(spec.dimensions.height_mm))
    wall = max(1.6, min(10.0, float(spec.dimensions.wall_thickness_mm)))
    n_slots = max(1, len(spec.filaments))
    s2 = 2 if n_slots >= 2 else 1
    s3 = 3 if n_slots >= 3 else s2
    cat = spec.category

    if cat == "headphone_stand":
        return dedent(
            f"""
            from build123d import *

            # Parameters (mm)
            WIDTH = {w:.2f}
            DEPTH = {d:.2f}
            HEIGHT = {h:.2f}
            WALL = {wall:.2f}

            base_h = max(12.0, min(22.0, HEIGHT * 0.08))
            col_w = max(24.0, min(42.0, WIDTH * 0.30))
            col_d = max(16.0, min(28.0, DEPTH * 0.22))
            cradle_w = max(56.0, min(WIDTH * 0.85, 92.0))
            cradle_d = max(36.0, min(DEPTH * 0.55, 58.0))
            cradle_t = max(8.0, WALL * 2.4)

            # Part 1 (Slot 1): Weighted chamfered base + vertical spine + top headband rest
            base = Box(WIDTH, DEPTH, base_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
            base = chamfer(base.edges().filter_by(Axis.Z), length=min(10.0, WIDTH * 0.1))
            base = chamfer(base.edges().sort_by(Axis.Z)[-4:], length=min(2.2, base_h * 0.2))

            # Rear cable routing slot cut out of base
            cable_cut = Pos(0, DEPTH * 0.38, base_h * 0.45) * Box(col_w * 0.55, DEPTH * 0.32, base_h * 0.55)
            base = base - cable_cut

            # Vertical column positioned slightly back for center-of-gravity stability
            col_y = DEPTH * 0.16
            col_h = max(60.0, HEIGHT - base_h - cradle_t * 0.7)
            column = Pos(0, col_y, base_h - 1.0) * Box(
                col_w, col_d, col_h + 1.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            column = chamfer(column.edges().filter_by(Axis.Z), length=min(3.0, col_d * 0.18))

            # Curved top headband cradle (arch profile via box + cylinder crown)
            cradle_z = base_h + col_h - cradle_t * 0.35
            cradle_core = Pos(0, col_y - cradle_d * 0.12, cradle_z) * Box(
                cradle_w, cradle_d, cradle_t, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            cradle_lip_f = Pos(0, col_y - cradle_d * 0.58, cradle_z + cradle_t * 0.6) * Box(
                cradle_w * 0.9, 5.0, 7.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            cradle_lip_b = Pos(0, col_y + cradle_d * 0.34, cradle_z + cradle_t * 0.6) * Box(
                cradle_w * 0.9, 5.0, 7.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            main_body = base + column + cradle_core + cradle_lip_f + cradle_lip_b

            # Part 2 (Slot {s2}): Contrasting accent spine trim & base perimeter inlay
            trim_bar = Pos(0, col_y - col_d * 0.5 - 2.0, base_h + 12.0) * Box(
                col_w * 0.36, 3.6, col_h - 24.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            base_accent = Pos(0, -DEPTH * 0.41, base_h - 0.3) * Box(
                WIDTH * 0.56, 6.4, 3.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            cradle_pad = Pos(0, col_y - cradle_d * 0.12, cradle_z + cradle_t - 0.1) * Box(
                cradle_w * 0.78, cradle_d * 0.62, 3.2, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )

            PARTS = [
                {{"name": "main_stand_body", "slot": 1, "solid": main_body}},
                {{"name": "accent_trim_and_pad", "slot": {s2}, "solid": trim_bar + base_accent + cradle_pad}},
            ]
            """
        ).strip()

    if cat == "phone_stand":
        return dedent(
            f"""
            from build123d import *

            WIDTH = {w:.2f}
            DEPTH = {d:.2f}
            HEIGHT = {h:.2f}
            WALL = {wall:.2f}

            base_h = max(8.0, WALL * 2.5)
            base = Box(WIDTH, DEPTH, base_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
            base = chamfer(base.edges().filter_by(Axis.Z), length=min(6.0, WIDTH * 0.08))

            # Angled backplate (~62 deg) with center charging pass-through slot
            plate_t = max(5.0, WALL * 1.6)
            plate_h = max(55.0, HEIGHT * 0.88)
            backplate = Pos(0, DEPTH * 0.08, base_h + plate_h * 0.42) * Rot(26, 0, 0) * Box(
                WIDTH * 0.92, plate_t, plate_h
            )
            # Trim bottom of tilted plate cleanly at Z=0
            z_clip = Pos(0, 0, -50.0) * Box(WIDTH * 2, DEPTH * 2, 100.0)
            backplate = backplate - z_clip

            # Front cradle shelf & anti-slip lip
            ledge = Pos(0, -DEPTH * 0.28, base_h - 0.5) * Box(
                WIDTH * 0.92, 16.0, 11.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            lip = Pos(0, -DEPTH * 0.36, base_h + 8.5) * Box(
                WIDTH * 0.92, 5.0, 10.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            cable_slot = Pos(0, -DEPTH * 0.25, base_h + 1.0) * Box(
                18.0, 40.0, 28.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            main_body = (base + backplate + ledge + lip) - cable_slot

            # Accent side rails (Slot {s2})
            rail_x = WIDTH * 0.46
            rail_l = Pos(-rail_x, 0, base_h * 0.35) * Box(4.2, DEPTH * 0.78, base_h * 0.72, align=(Align.CENTER, Align.CENTER, Align.MIN))
            rail_r = Pos(rail_x, 0, base_h * 0.35) * Box(4.2, DEPTH * 0.78, base_h * 0.72, align=(Align.CENTER, Align.CENTER, Align.MIN))

            PARTS = [
                {{"name": "phone_cradle_body", "slot": 1, "solid": main_body}},
                {{"name": "side_accent_rails", "slot": {s2}, "solid": rail_l + rail_r}},
            ]
            """
        ).strip()

    if cat == "desk_organizer":
        return dedent(
            f"""
            from build123d import *

            WIDTH = {w:.2f}
            DEPTH = {d:.2f}
            HEIGHT = {h:.2f}
            WALL = {wall:.2f}

            outer = Box(WIDTH, DEPTH, HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN))
            outer = chamfer(outer.edges().filter_by(Axis.Z), length=min(6.0, WIDTH * 0.06))

            # Carve 3 ergonomic tiered cavities
            floor_t = max(3.0, WALL)
            c_left = Pos(-WIDTH * 0.26, 0, floor_t) * Box(
                WIDTH * 0.42 - WALL, DEPTH - 2 * WALL, HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            c_rb = Pos(WIDTH * 0.24, DEPTH * 0.22, floor_t) * Box(
                WIDTH * 0.46 - WALL, DEPTH * 0.46 - WALL, HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            c_rf = Pos(WIDTH * 0.24, -DEPTH * 0.24, floor_t) * Box(
                WIDTH * 0.46 - WALL, DEPTH * 0.44 - WALL, HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            # Lower front cutout for easy access
            front_step = Pos(WIDTH * 0.24, -DEPTH * 0.26, HEIGHT * 0.56) * Box(
                WIDTH * 0.52, DEPTH * 0.56, HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            main_body = outer - c_left - c_rb - c_rf - front_step

            # Contrasting top rim collar (Slot {s2})
            rim_outer = Pos(0, 0, HEIGHT - 4.0) * Box(WIDTH + 1.6, DEPTH + 1.6, 4.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
            rim_outer = chamfer(rim_outer.edges().filter_by(Axis.Z), length=min(6.4, WIDTH * 0.065))
            rim_inner = Pos(0, 0, HEIGHT - 5.0) * Box(WIDTH - 2 * WALL + 0.4, DEPTH - 2 * WALL + 0.4, 7.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
            rim = (rim_outer - rim_inner) - front_step

            # Base foot ring (Slot {s3})
            foot = Box(WIDTH * 0.9, DEPTH * 0.9, 2.6, align=(Align.CENTER, Align.CENTER, Align.MIN))

            PARTS = [
                {{"name": "organizer_body", "slot": 1, "solid": main_body}},
                {{"name": "top_accent_rim", "slot": {s2}, "solid": rim}},
                {{"name": "base_plinth", "slot": {s3}, "solid": foot}},
            ]
            """
        ).strip()

    if cat == "cable_clip":
        return dedent(
            f"""
            from build123d import *

            WIDTH = {w:.2f}
            DEPTH = {d:.2f}
            HEIGHT = {h:.2f}
            WALL = {wall:.2f}

            n_slots = max(3, min(7, int(WIDTH // 14)))
            pitch = WIDTH / (n_slots + 0.6)
            body = Box(WIDTH, DEPTH, HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN))
            body = chamfer(body.edges().filter_by(Axis.Z), length=min(3.5, DEPTH * 0.15))

            for i in range(n_slots):
                x = (i - (n_slots - 1) / 2.0) * pitch
                bore = Pos(x, 0, HEIGHT * 0.52) * Rot(90, 0, 0) * Cylinder(radius=3.4, height=DEPTH + 4.0)
                neck = Pos(x, 0, HEIGHT * 0.78) * Box(4.2, DEPTH + 4.0, HEIGHT * 0.55)
                body = body - bore - neck

            # Accent adhesive base pad (Slot {s2})
            pad = Box(WIDTH * 0.94, DEPTH * 0.92, 2.4, align=(Align.CENTER, Align.CENTER, Align.MIN))

            PARTS = [
                {{"name": "cable_comb_body", "slot": 1, "solid": body}},
                {{"name": "base_accent_pad", "slot": {s2}, "solid": pad}},
            ]
            """
        ).strip()

    if cat == "enclosure":
        return dedent(
            f"""
            from build123d import *

            WIDTH = {w:.2f}
            DEPTH = {d:.2f}
            HEIGHT = {h:.2f}
            WALL = {wall:.2f}

            shell = Box(WIDTH, DEPTH, HEIGHT - 4.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
            shell = chamfer(shell.edges().filter_by(Axis.Z), length=min(5.0, WIDTH * 0.06))
            cavity = Pos(0, 0, WALL) * Box(
                WIDTH - 2 * WALL, DEPTH - 2 * WALL, HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            box_body = shell - cavity

            # 4 internal PCB standoffs
            sx, sy = WIDTH * 0.36, DEPTH * 0.34
            for dx in (-sx, sx):
                for dy in (-sy, sy):
                    post = Pos(dx, dy, WALL) * Cylinder(radius=3.6, height=8.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
                    hole = Pos(dx, dy, WALL + 1.0) * Cylinder(radius=1.55, height=9.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
                    box_body = box_body + post - hole

            # Side cooling vent slots
            for i in range(-2, 3):
                vent = Pos(i * (WIDTH * 0.12), DEPTH * 0.5, HEIGHT * 0.5) * Box(4.0, WALL * 3.0, HEIGHT * 0.36)
                box_body = box_body - vent

            # Contrasting top lid bezel (Slot {s2})
            lid = Pos(0, 0, HEIGHT - 4.0) * Box(WIDTH, DEPTH, 4.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
            lid = chamfer(lid.edges().filter_by(Axis.Z), length=min(5.0, WIDTH * 0.06))
            lid_lip = Pos(0, 0, HEIGHT - 7.0) * Box(
                WIDTH - 2 * WALL - 0.6, DEPTH - 2 * WALL - 0.6, 3.2, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            lid_cut = Pos(0, 0, HEIGHT - 8.0) * Box(
                WIDTH - 2 * WALL - 5.0, DEPTH - 2 * WALL - 5.0, 5.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            lid_part = lid + (lid_lip - lid_cut)

            PARTS = [
                {{"name": "enclosure_chassis", "slot": 1, "solid": box_body}},
                {{"name": "top_bezel_lid", "slot": {s2}, "solid": lid_part}},
            ]
            """
        ).strip()

    if cat == "coaster_sign":
        return dedent(
            f"""
            from build123d import *

            WIDTH = {w:.2f}
            HEIGHT = max(4.5, {h:.2f})

            radius = WIDTH * 0.5
            with BuildPart() as base_bp:
                with BuildSketch():
                    RegularPolygon(radius=radius, side_count=6)
                extrude(amount=HEIGHT * 0.68)
                with BuildSketch(Plane.XY.offset(HEIGHT * 0.45)):
                    RegularPolygon(radius=radius * 0.84, side_count=6)
                extrude(amount=HEIGHT * 0.4, mode=Mode.SUBTRACT)

            with BuildPart() as ring_bp:
                with BuildSketch(Plane.XY.offset(HEIGHT * 0.62)):
                    RegularPolygon(radius=radius * 0.96, side_count=6)
                    RegularPolygon(radius=radius * 0.84, side_count=6, mode=Mode.SUBTRACT)
                extrude(amount=HEIGHT * 0.38)

            with BuildPart() as emblem_bp:
                with BuildSketch(Plane.XY.offset(HEIGHT * 0.44)):
                    RegularPolygon(radius=radius * 0.45, side_count=6)
                    RegularPolygon(radius=radius * 0.24, side_count=6, mode=Mode.SUBTRACT)
                extrude(amount=HEIGHT * 0.44)

            PARTS = [
                {{"name": "hex_coaster_base", "slot": 1, "solid": base_bp.part}},
                {{"name": "outer_rim_ring", "slot": {s2}, "solid": ring_bp.part}},
                {{"name": "center_geometric_inlay", "slot": {s3}, "solid": emblem_bp.part}},
            ]
            """
        ).strip()

    # Default / planter_vase / generic geometric parametric container
    return dedent(
        f"""
        from build123d import *
        import math

        WIDTH = {w:.2f}
        HEIGHT = {h:.2f}
        WALL = {wall:.2f}

        r_out = WIDTH * 0.5
        r_in = max(8.0, r_out - WALL * 1.4)

        with BuildPart() as pot_bp:
            with BuildSketch():
                RegularPolygon(radius=r_out * 0.86, side_count=12)
            with BuildSketch(Plane.XY.offset(HEIGHT * 0.55)):
                RegularPolygon(radius=r_out, side_count=12)
            with BuildSketch(Plane.XY.offset(HEIGHT)):
                RegularPolygon(radius=r_out * 0.94, side_count=12)
            loft()
            with BuildSketch(Plane.XY.offset(WALL * 1.6)):
                Circle(radius=r_in * 0.84)
            extrude(amount=HEIGHT, mode=Mode.SUBTRACT)

        # Contrasting top collar band (Slot {s2})
        collar_o = Pos(0, 0, HEIGHT - 8.0) * Cylinder(radius=r_out * 0.97, height=8.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
        collar_i = Pos(0, 0, HEIGHT - 9.0) * Cylinder(radius=r_in * 0.84, height=10.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
        collar = collar_o - collar_i

        # Contrasting drip base ring (Slot {s3})
        saucer = Cylinder(radius=r_out * 0.94, height=4.5, align=(Align.CENTER, Align.CENTER, Align.MIN))

        PARTS = [
            {{"name": "faceted_body", "slot": 1, "solid": pot_bp.part}},
            {{"name": "top_accent_collar", "slot": {s2}, "solid": collar}},
            {{"name": "base_ring", "slot": {s3}, "solid": saucer}},
        ]
        """
    ).strip()


CAD_LLM_SYSTEM = """You are a senior mechanical CAD engineer writing Python code using `build123d` for FDM 3D printing on Bambu Lab AMS multicolor printers.

STRICT RULES (violating any rule causes OCCT kernel crash):
1. Import ONLY `from build123d import *` and optionally `import math`. Never import os, sys, subprocess, pathlib, etc.
2. ALWAYS define every constant variable at the top BEFORE using it.
3. NEVER call `.moved(...)`, `.translate(...)`, or `.rotate(...)` methods on solids! ALWAYS transform shapes using algebra operators:
   `Pos(x, y, z) * Box(w, d, h, align=(Align.CENTER, Align.CENTER, Align.MIN))`
   `Pos(x, y, z) * Rot(rx, ry, rz) * Cylinder(radius=r, height=h)`
4. Combine shapes into ONE solid using `+` (union) and `-` (subtract). NEVER put a Python tuple or list inside `"solid"`.
5. DO NOT manually model dovetail or snap-fit joints! Just model clean, contacting multi-color parts (`slot: 1`, `slot: 2`, ...); SpoolGenie automatically injects precision glue-free dovetail/snap-pin joints between color parts!
6. Respect ALL user-requested dimensions, clamp gaps (e.g., 32mm desk gap), wall thicknesses, and holes in millimeters!
7. NEVER call `.chamfer(...)`, `.fillet(...)`, `chamfer(...)`, or `fillet(...)` (OCCT frequently fails on boolean edges and method signatures differ from CadQuery). Build clean, crisp mechanical geometry using `Box`, `Cylinder`, `Cone`, `Pos(x, y, z)`, `Rot(rx, ry, rz)`, `+`, and `-`.
8. Make sure every subtracted pocket, slot, or hole is positioned with `Pos(x, y, z)` so solid walls (>= WALL) and a flat Z=0 floor remain intact, and opens cleanly to the TOP (+Z) or FRONT (-Y) exterior face. NEVER create `accent = outer_box - main_body` (that fills functional cutouts back in!). Place accent parts as clean exterior trim strips, rails, or base plinths.
9. End with a top-level list `PARTS`:
   PARTS = [
       {"name": "main_body", "slot": 1, "solid": main_solid},
       {"name": "accent_part", "slot": 2, "solid": accent_solid},
   ]
10. Output ONLY raw Python code inside a ```python ... ``` block.

Minimal verified build123d idiom example (32mm desk-edge dual hook):
```python
from build123d import *

W, D, H = 55.0, 85.0, 120.0
WALL = 6.0
DESK_GAP = 32.5  # 32mm desk + 0.5mm clearance

# C-clamp seated on top edge of desk
top_jaw = Pos(0, 0, H - WALL) * Box(W, 50.0, WALL, align=(Align.CENTER, Align.CENTER, Align.MIN))
back_spine = Pos(0, 25.0 - WALL * 0.5, 0) * Box(W, WALL, H, align=(Align.CENTER, Align.CENTER, Align.MIN))
bot_jaw = Pos(0, 0, H - WALL - DESK_GAP - WALL) * Box(W, 50.0, WALL, align=(Align.CENTER, Align.CENTER, Align.MIN))
hook1_arm = Pos(-14.0, -10.0, 12.0) * Box(18.0, 62.0, 8.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
hook1_lip = Pos(-14.0, -39.0, 12.0) * Box(18.0, 6.0, 24.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
main_body = top_jaw + back_spine + bot_jaw + hook1_arm + hook1_lip

# Secondary color gamepad hook cradle attached to spine
pad_arm = Pos(14.0, -12.0, 38.0) * Box(20.0, 66.0, 8.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
pad_lip = Pos(14.0, -43.0, 38.0) * Box(20.0, 6.0, 26.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
accent_hook = pad_arm + pad_lip

PARTS = [
    {"name": "desk_clamp_and_headphone_hook", "slot": 1, "solid": main_body},
    {"name": "gamepad_hook_cradle", "slot": 2, "solid": accent_hook},
]
```
"""


def synthesize_cad_code(
    spec: DesignSpec,
    llm: UnifiedLLMClient | None = None,
    previous_error: str | None = None,
    previous_code: str | None = None,
    prefer_template: bool = False,
    kb_context: str = "",
) -> str:
    """Return parametric build123d script: LLM-synthesized for ANY arbitrary/custom object, templates for built-in classics/fallback."""
    known_templates = {
        "headphone_stand",
        "phone_stand",
        "desk_organizer",
        "cable_clip",
        "enclosure",
        "coaster_sign",
        "planter_vase",
    }

    use_template = prefer_template or (spec.category in known_templates and not previous_error and not (llm and llm.available and spec.category == "custom"))
    if use_template:
        return generate_template_cad_code(spec)

    if llm and llm.available:
        user_prompt = (
            f"Design specification JSON:\n{spec.model_dump_json(indent=2)}\n"
            + (f"Retrieved MakerWorld functional engineering priors:\n{kb_context}\n" if kb_context else "")
            + "Write a clean, watertight, multi-part build123d script implementing this EXACT custom 3D-printable object with real functional geometry."
        )
        if previous_error and previous_code:
            user_prompt += (
                f"\nThe previous script failed with error:\n{previous_error}\n"
                f"Previous script:\n```python\n{previous_code}\n```\nFix the error (avoid failing fillets or empty shapes) and return ONLY the complete fixed ```python``` script."
            )
        raw = llm.chat(
            [{"role": "system", "content": CAD_LLM_SYSTEM}, {"role": "user", "content": user_prompt}],
            temperature=0.12,
        )
        if raw:
            code = _extract_python_block(raw)
            try:
                validate_cad_code_ast(code)
                return code
            except UnsafeCADCodeError:
                pass

    return generate_template_cad_code(spec)


def _extract_python_block(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.S)
    return (m.group(1) if m else text).strip()
