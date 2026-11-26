"""Bambu Studio & OrcaSlicer spec-compliant multi-color .3mf, STL, and Three.js GLB exporter.

Supports BOTH native Bambu Lab multi-color representations:
  1. Multi-body assembly .3mf (`Metadata/model_settings.config` extruder=1..16 + `Metadata/project_settings.config`)
  2. Single-manifold per-triangle painted .3mf (`paint_color` nibble encoding: 1->"4", 2->"8", 3->"0C", 4->"1C", ..., 16->"DC")
  3. Separate single-color .stl + laid-out split-plate .3mf for No-AMS glue-free mechanical assembly!
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import trimesh

from spoolgenie.print_prep.split_joiner import PrintPart
from spoolgenie.schemas import FilamentSlot


def encode_bambu_paint_color(slot_1based: int) -> str:
    """Encode a 1-based AMS filament slot (1..16) into Bambu Studio / PrusaSlicer whole-triangle `paint_color`.

    Verified spec (nibbles read right-to-left, low 2 bits = split_mode 0b00):
      - slot 1 -> "4"  (0b0100)
      - slot 2 -> "8"  (0b1000)
      - slot 3 -> "0C" (extended escape 0b1100 + 0)
      - slot 4 -> "1C" (extended escape 0b1100 + 1)
      - slot k (3..16) -> f"{k - 3:X}C"
    """
    slot = max(1, min(16, int(slot_1based)))
    if slot == 1:
        return "4"
    if slot == 2:
        return "8"
    return f"{slot - 3:X}C"


def _hex_to_rgba(hx: str) -> tuple[int, int, int, int]:
    s = hx.lstrip("#")
    if len(s) != 6:
        return (43, 45, 48, 255)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255)


def _format_mesh_xml(
    mesh: trimesh.Trimesh,
    face_slots: np.ndarray | None = None,
    slot_to_color_idx: dict[int, int] | None = None,
    color_group_id: int | None = None,
) -> str:
    """Serialize vertices and triangles into 3MF <mesh> XML with optional Bambu paint_color & 3MF material color."""
    v_lines = [
        f'          <vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>'
        for v in mesh.vertices
    ]
    t_lines: list[str] = []
    if face_slots is not None and len(face_slots) == len(mesh.faces):
        for tri, slot in zip(mesh.faces, face_slots):
            s_int = int(slot)
            pc = encode_bambu_paint_color(s_int)
            extra = f' paint_color="{pc}"'
            if color_group_id is not None and slot_to_color_idx is not None:
                cidx = slot_to_color_idx.get(s_int, 0)
                extra += f' pid="{color_group_id}" p1="{cidx}"'
            t_lines.append(f'          <triangle v1="{int(tri[0])}" v2="{int(tri[1])}" v3="{int(tri[2])}"{extra}/>')
    else:
        for tri in mesh.faces:
            t_lines.append(f'          <triangle v1="{int(tri[0])}" v2="{int(tri[1])}" v3="{int(tri[2])}"/>')

    return (
        "        <mesh>\n"
        "          <vertices>\n"
        + "\n".join(v_lines)
        + "\n          </vertices>\n"
        "          <triangles>\n"
        + "\n".join(t_lines)
        + "\n          </triangles>\n"
        "        </mesh>"
    )


CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
  <Default Extension="config" ContentType="text/xml"/>
  <Default Extension="png" ContentType="image/png"/>
</Types>
"""

ROOT_RELS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def _build_project_settings_json(
    filaments: list[FilamentSlot],
    wall_loops: int = 3,
    sparse_infill_density: str = "15%",
    material_override: str | None = None,
) -> str:
    cols = [f.hex_color.upper() for f in filaments] or ["#2B2D30"]
    types = [
        (material_override.upper() if material_override else f.material.upper()) for f in filaments
    ] or [(material_override.upper() if material_override else "PLA")]
    payload = {
        "from": "SpoolGenie",
        "filament_colour": cols,
        "filament_type": types,
        "filament_settings_id": [f"Bambu {t} Basic" for t in types],
        "layer_height": "0.2",
        "sparse_infill_density": sparse_infill_density,
        "sparse_infill_pattern": "gyroid",
        "wall_loops": str(max(2, int(wall_loops))),
        "enable_prime_tower": "1" if len(cols) > 1 else "0",
    }
    return json.dumps(payload, indent=2)


def export_multipart_bambu_3mf(
    parts: list[PrintPart],
    filaments: list[FilamentSlot],
    output_path: str | Path,
    model_title: str = "SpoolGenie Assembly",
    as_separate_objects: bool = False,
    wall_loops: int = 3,
    sparse_infill_density: str = "15%",
) -> Path:
    """Export multi-body Bambu Studio / OrcaSlicer .3mf with exact per-part AMS extruder assignments.

    If `as_separate_objects=False` (AMS assembly): all parts belong to one unified assembly object so
    they stay locked in place on the bed while each sub-part uses its assigned AMS slot.
    If `as_separate_objects=True` (No-AMS split print): each part is an independent bed object ready to
    print color-by-color or side-by-side without glue.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    slot_map = {f.slot: idx for idx, f in enumerate(filaments)}
    color_entries = "\n".join(
        f'        <m:color color="{f.hex_color.upper()}FF"/>' for f in (filaments or [FilamentSlot(slot=1)])
    )

    object_blocks: list[str] = []
    part_settings_blocks: list[str] = []
    build_items: list[str] = []

    for idx, part in enumerate(parts, start=1):
        obj_id = idx + 1  # reserve id=1 for colorgroup
        c_idx = slot_map.get(part.slot, 0)
        mesh_xml = _format_mesh_xml(part.mesh)
        object_blocks.append(
            f'      <object id="{obj_id}" name="{escape(part.name)}" type="model" pid="1" pindex="{c_idx}">\n'
            f"{mesh_xml}\n"
            f"      </object>"
        )
        part_settings_blocks.append(
            f'    <part id="{obj_id}" subtype="normal_part">\n'
            f'      <metadata key="name" value="{escape(part.name)}"/>\n'
            f'      <metadata key="extruder" value="{part.slot}"/>\n'
            f'      <metadata key="source_object_id" value="0"/>\n'
            f'      <metadata key="source_volume_id" value="{idx - 1}"/>\n'
            f"    </part>"
        )

    if as_separate_objects or len(parts) == 1:
        for idx, part in enumerate(parts, start=1):
            obj_id = idx + 1
            build_items.append(f'    <item objectid="{obj_id}" printable="1"/>')
        model_settings_objects = "\n".join(
            f'  <object id="{idx + 1}">\n'
            f'    <metadata key="name" value="{escape(part.name)}"/>\n'
            f'    <metadata key="extruder" value="{part.slot}"/>\n'
            f"  </object>"
            for idx, part in enumerate(parts, start=1)
        )
    else:
        assembly_id = len(parts) + 2
        comp_lines = "\n".join(
            f'          <component objectid="{idx + 1}"/>' for idx in range(1, len(parts) + 1)
        )
        object_blocks.append(
            f'      <object id="{assembly_id}" name="{escape(model_title)}" type="model">\n'
            f"        <components>\n{comp_lines}\n        </components>\n"
            f"      </object>"
        )
        build_items.append(f'    <item objectid="{assembly_id}" printable="1"/>')
        model_settings_objects = (
            f'  <object id="{assembly_id}">\n'
            f'    <metadata key="name" value="{escape(model_title)}"/>\n'
            f'    <metadata key="extruder" value="1"/>\n'
            + "\n".join(part_settings_blocks)
            + "\n  </object>"
        )

    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02" '
        'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">\n'
        '  <metadata name="Application">SpoolGenie 0.1.0</metadata>\n'
        f'  <metadata name="Title">{escape(model_title)}</metadata>\n'
        "  <resources>\n"
        '    <m:colorgroup id="1">\n'
        f"{color_entries}\n"
        "    </m:colorgroup>\n"
        + "\n".join(object_blocks)
        + "\n  </resources>\n"
        "  <build>\n"
        + "\n".join(build_items)
        + "\n  </build>\n"
        "</model>\n"
    )

    model_settings_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<config>\n"
        f"{model_settings_objects}\n"
        "</config>\n"
    )

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        zf.writestr("_rels/.rels", ROOT_RELS_XML)
        zf.writestr("3D/3dmodel.model", model_xml)
        zf.writestr("Metadata/model_settings.config", model_settings_xml)
        zf.writestr(
            "Metadata/project_settings.config",
            _build_project_settings_json(
                filaments,
                wall_loops=wall_loops,
                sparse_infill_density=sparse_infill_density,
            ),
        )

    return out


def export_painted_bambu_3mf(
    mesh: trimesh.Trimesh,
    face_slots: np.ndarray,
    filaments: list[FilamentSlot],
    output_path: str | Path,
    model_title: str = "SpoolGenie Painted Model",
    wall_loops: int = 3,
    sparse_infill_density: str = "15%",
) -> Path:
    """Export a single watertight manifold mesh with native Bambu Studio per-triangle `paint_color` tags."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    slot_map = {f.slot: idx for idx, f in enumerate(filaments)}
    color_entries = "\n".join(
        f'        <m:color color="{f.hex_color.upper()}FF"/>' for f in (filaments or [FilamentSlot(slot=1)])
    )
    mesh_xml = _format_mesh_xml(mesh, face_slots=face_slots, slot_to_color_idx=slot_map, color_group_id=1)

    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02" '
        'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">\n'
        '  <metadata name="Application">SpoolGenie 0.1.0</metadata>\n'
        f'  <metadata name="Title">{escape(model_title)}</metadata>\n'
        "  <resources>\n"
        '    <m:colorgroup id="1">\n'
        f"{color_entries}\n"
        "    </m:colorgroup>\n"
        f'    <object id="2" name="{escape(model_title)}" type="model">\n'
        f"{mesh_xml}\n"
        "    </object>\n"
        "  </resources>\n"
        "  <build>\n"
        '    <item objectid="2" printable="1"/>\n'
        "  </build>\n"
        "</model>\n"
    )

    model_settings_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<config>\n"
        '  <object id="2">\n'
        f'    <metadata key="name" value="{escape(model_title)}"/>\n'
        '    <metadata key="extruder" value="1"/>\n'
        "  </object>\n"
        "</config>\n"
    )

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        zf.writestr("_rels/.rels", ROOT_RELS_XML)
        zf.writestr("3D/3dmodel.model", model_xml)
        zf.writestr("Metadata/model_settings.config", model_settings_xml)
        zf.writestr(
            "Metadata/project_settings.config",
            _build_project_settings_json(
                filaments,
                wall_loops=wall_loops,
                sparse_infill_density=sparse_infill_density,
            ),
        )

    return out


def export_colored_preview_glb(
    parts: list[PrintPart],
    filaments: list[FilamentSlot],
    output_path: str | Path,
    face_slots_override: np.ndarray | None = None,
) -> Path:
    """Export a clean PBR colored GLB for instant Three.js interactive preview in the Web UI."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    slot_rgba = {f.slot: _hex_to_rgba(f.hex_color) for f in filaments}
    default_rgba = _hex_to_rgba(filaments[0].hex_color) if filaments else (43, 45, 48, 255)

    scene = trimesh.Scene()
    for idx, part in enumerate(parts):
        m = part.mesh.copy()
        if face_slots_override is not None and len(parts) == 1 and len(face_slots_override) == len(m.faces):
            # Split triangles by color slot into separate clean PBR submeshes so Three.js renders crisp matte colors
            for slot_val in sorted(set(int(s) for s in face_slots_override)):
                mask = face_slots_override == slot_val
                sub = m.submesh([np.where(mask)[0]], append=True)
                if isinstance(sub, trimesh.Trimesh) and len(sub.faces) > 0:
                    rgba = slot_rgba.get(slot_val, default_rgba)
                    sub.visual = trimesh.visual.TextureVisuals(
                        material=trimesh.visual.material.PBRMaterial(
                            baseColorFactor=rgba,
                            metallicFactor=0.05,
                            roughnessFactor=0.68,
                        )
                    )
                    scene.add_geometry(sub, node_name=f"{part.name}_slot{slot_val}")
        else:
            rgba = slot_rgba.get(part.slot, default_rgba)
            m.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=rgba,
                    metallicFactor=0.05,
                    roughnessFactor=0.68,
                )
            )
            scene.add_geometry(m, node_name=f"{part.name}_{idx+1}")

    scene.export(out, file_type="glb")
    return out
