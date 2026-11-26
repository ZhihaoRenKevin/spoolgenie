"""Tests for Bambu Studio / OrcaSlicer multi-color .3mf packaging and paint_color bit encoding."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from spoolgenie.print_prep.bambu_3mf import (
    encode_bambu_paint_color,
    export_multipart_bambu_3mf,
    export_painted_bambu_3mf,
)
from spoolgenie.print_prep.split_joiner import PrintPart
from spoolgenie.schemas import FilamentSlot


def test_bambu_paint_color_encoding_spec() -> None:
    assert encode_bambu_paint_color(1) == "4"
    assert encode_bambu_paint_color(2) == "8"
    assert encode_bambu_paint_color(3) == "0C"
    assert encode_bambu_paint_color(4) == "1C"
    assert encode_bambu_paint_color(5) == "2C"
    assert encode_bambu_paint_color(16) == "DC"


def test_export_multipart_bambu_3mf(tmp_path: Path) -> None:
    b1 = trimesh.creation.box(extents=[40, 40, 10])
    b2 = trimesh.creation.box(extents=[20, 20, 8])
    b2.apply_translation([0, 0, 9])

    parts = [
        PrintPart(name="chassis", slot=1, mesh=b1),
        PrintPart(name="accent_cap", slot=2, mesh=b2),
    ]
    filaments = [
        FilamentSlot(slot=1, name="Matte Charcoal", hex_color="#2B2D30"),
        FilamentSlot(slot=2, name="Matte Mandarin Orange", hex_color="#F36C21"),
    ]
    out = export_multipart_bambu_3mf(parts, filaments, tmp_path / "assembly.3mf", model_title="TestStand")
    assert out.exists()

    with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())
        assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model", "Metadata/model_settings.config", "Metadata/project_settings.config"} <= names
        model_xml = zf.read("3D/3dmodel.model").decode("utf-8")
        ET.fromstring(model_xml)
        assert "<components>" in model_xml

        settings_xml = zf.read("Metadata/model_settings.config").decode("utf-8")
        assert 'key="extruder" value="2"' in settings_xml

        proj = json.loads(zf.read("Metadata/project_settings.config").decode("utf-8"))
        assert proj["filament_colour"] == ["#2B2D30", "#F36C21"]


def test_export_painted_bambu_3mf(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=20.0)
    slots = np.array([(i % 4) + 1 for i in range(len(mesh.faces))], dtype=np.int32)
    filaments = [
        FilamentSlot(slot=1, name="Black", hex_color="#2B2D30"),
        FilamentSlot(slot=2, name="Orange", hex_color="#F36C21"),
        FilamentSlot(slot=3, name="White", hex_color="#F2EFE9"),
        FilamentSlot(slot=4, name="Blue", hex_color="#1E4C8A"),
    ]
    out = export_painted_bambu_3mf(mesh, slots, filaments, tmp_path / "painted.3mf")
    with zipfile.ZipFile(out, "r") as zf:
        xml_txt = zf.read("3D/3dmodel.model").decode("utf-8")
        ET.fromstring(xml_txt)
        for code in ('paint_color="4"', 'paint_color="8"', 'paint_color="0C"', 'paint_color="1C"'):
            assert code in xml_txt
