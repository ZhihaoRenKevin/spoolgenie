"""Core Pydantic schemas shared across CLI, Python API, and Web UI."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EngineRoute(str, Enum):
    CAD = "cad"          # Parametric build123d functional/mechanical part
    MESH = "mesh"        # Organic / sculptural 2D -> 3D mesh generation
    HYBRID = "hybrid"    # Parametric functional base + sculpted accent


class MulticolorMode(str, Enum):
    AUTO = "auto"                # AMS if ams_units > 0, else glue-free split_no_ams when >1 color
    AMS = "ams"                  # Single-print multi-material via Bambu AMS (1..4 units -> 4..16 colors)
    SPLIT_NO_AMS = "split_no_ams"  # Separate single-color interlocking parts with glue-free mechanical joints
    MONOCHROME = "monochrome"    # Single filament unified solid


class JointType(str, Enum):
    DOVETAIL = "dovetail"        # Sliding tapered dovetail (strong, glue-free, anti-pullout)
    SNAP_PIN = "snap_pin"        # Slotted compliant snap-fit peg + chamfered socket + anti-rotation pin
    PRESS_TENON = "press_tenon"  # Chamfered friction-fit mortise & tenon
    NONE = "none"                # No mechanical joints added


class WorkflowStage(str, Enum):
    IDEATION = "ideation"            # Collecting prompt & MakerWorld references
    CONCEPT_REVIEW = "concept_review"  # 2D preview + color/dimension confirmation gate
    MODEL_READY = "model_ready"      # 3D mesh/CAD generated & DFM checked
    EXPORTED = "exported"            # Bambu AMS 3MF / STL / STEP / optional sliced G-code ready


class FilamentSlot(BaseModel):
    slot: int = Field(ge=1, le=16, description="1-based Bambu AMS extruder slot")
    name: str = Field(default="Matte Charcoal", description="Human-readable filament color name")
    hex_color: str = Field(default="#2B2D30", pattern=r"^#[0-9A-Fa-f]{6}$")
    material: str = Field(default="PLA", description="PLA, PETG, ABS, TPU, etc.")
    role: str = Field(default="primary", description="Where this filament is used (body, accent, text, base)")


class DimensionsMM(BaseModel):
    width_mm: float = Field(default=110.0, gt=5.0, le=500.0)
    depth_mm: float = Field(default=90.0, gt=5.0, le=500.0)
    height_mm: float = Field(default=180.0, gt=2.0, le=500.0)
    wall_thickness_mm: float = Field(default=3.0, ge=0.8, le=30.0)
    clearance_mm: float = Field(default=0.25, ge=0.0, le=5.0)


class ReferenceAsset(BaseModel):
    source: Literal["makerworld_search", "makerworld_url", "upload_image", "upload_mesh"]
    title: str
    url: str | None = None
    local_path: str | None = None
    thumbnail_url: str | None = None
    snippet: str = ""
    extracted_style: list[str] = Field(default_factory=list)
    extracted_colors: list[str] = Field(default_factory=list)
    extracted_notes: str = ""


class FunctionalOp(BaseModel):
    op: Literal[
        "hollow_cavity",
        "phone_slot",
        "magsafe_recess",
        "magnet_sockets",
        "cable_comb",
        "plinth_base",
        "keychain_loop",
    ]
    params: dict[str, float | int | str | bool] = Field(default_factory=dict)


class DesignSpec(BaseModel):
    title: str = "Custom 3D Print"
    summary: str = ""
    category: str = "custom"
    route: EngineRoute = EngineRoute.CAD
    printer_id: str = "bambu_p1s"
    ams_units: int = Field(default=1, ge=0, le=4, description="Number of Bambu AMS units (0=no AMS, 1=4 colors, ..., 4=16 colors)")
    multicolor_mode: MulticolorMode = MulticolorMode.AUTO
    joint_type: JointType = JointType.DOVETAIL
    joint_clearance_mm: float = Field(default=0.22, ge=0.05, le=0.8, description="Mechanical joint printing clearance (mm)")
    dimensions: DimensionsMM = Field(default_factory=DimensionsMM)
    filaments: list[FilamentSlot] = Field(
        default_factory=lambda: [
            FilamentSlot(slot=1, name="Matte Charcoal", hex_color="#2B2D30", role="main body"),
            FilamentSlot(slot=2, name="Matte Mandarin Orange", hex_color="#F36C21", role="accent trim"),
        ]
    )
    owned_filaments: list[FilamentSlot] = Field(
        default_factory=list,
        description="Optional list of filaments the user already owns; canon colors match these first via CIELAB ΔE before recommending Bambu Lab official store refills.",
    )
    coloring_mode: Literal["modular", "surface_paint"] = Field(
        default="modular",
        description="'modular' decomposes meshes into watertight whole-color solid modules (recommended); 'surface_paint' uses legacy per-triangle painting.",
    )
    functional_features: list[str] = Field(default_factory=list)
    functional_ops: list[FunctionalOp] = Field(default_factory=list)
    style_keywords: list[str] = Field(default_factory=lambda: ["clean industrial", "chamfered edges"])
    prompt_2d: str = ""
    parametric_params: dict[str, float | int | str | bool] = Field(default_factory=dict)
    confirmed_by_user: bool = False


class ColorDecision(BaseModel):
    source_hex: str
    mapped_slot: int
    mapped_name: str
    mapped_hex: str
    delta_e: float = 0.0
    action: Literal["kept", "merged", "monochrome_collapsed"] = "kept"
    reason: str = ""


class FilamentMatchRecommendation(BaseModel):
    slot: int
    role: str = ""
    canon_name: str
    canon_hex: str
    status: Literal["canon_default", "owned_match", "purchase_recommended"] = "canon_default"
    matched_owned_name: str | None = None
    matched_owned_hex: str | None = None
    delta_e: float = 0.0
    bambu_name: str = ""
    bambu_name_zh: str = ""
    bambu_hex: str = "#2B2D30"
    bambu_series: str = "PLA Matte"
    store_url_cn: str = "https://store.bambulab.cn/products/pla-matte"
    store_url_global: str = "https://store.bambulab.com/products/pla-matte"
    reason: str = ""


class SplitPartInfo(BaseModel):
    name: str
    slot: int
    filament_name: str
    hex_color: str
    joint_type: str
    bbox_mm: tuple[float, float, float]
    stl_path: str | None = None
    plate_index: int = 1


class Concept2DAsset(BaseModel):
    backend: str
    image_path: str
    image_url: str | None = None
    prompt_used: str
    width: int = 1024
    height: int = 768
    notes: str = ""


class DFMCheckItem(BaseModel):
    name: str
    status: Literal["pass", "warn", "fail"]
    detail: str


class PhysicsCheckItem(BaseModel):
    name: str
    name_zh: str
    status: Literal["pass", "warn", "fail"]
    metric: str
    detail: str


class PhysicsStabilityReport(BaseModel):
    material_code: str = "PLA"
    material_name: str = "Bambu PLA Basic / Matte"
    density_g_cm3: float = 1.24
    youngs_modulus_gpa: float = 2.6
    tensile_xy_mpa: float = 46.0
    tensile_z_mpa: float = 24.0
    hdt_c: float = 55.0
    load_scenario: str = "Desktop Display"
    load_scenario_zh: str = "桌面摆件工况"
    payload_mass_g: float = 0.0
    self_mass_g: float = 0.0
    center_of_mass_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    support_area_mm2: float = 0.0
    flat_bottom_area_mm2: float = 0.0
    base_contact_ratio: float = 0.0
    base_closed_solid: bool = True
    tipping_margin_mm: float = 0.0
    tilt_angle_unloaded_deg: float = 0.0
    tilt_angle_loaded_deg: float = 0.0
    tipping_force_n: float = 0.0
    weakest_neck_z_mm: float = 0.0
    weakest_neck_diam_mm: float = 0.0
    peak_z_stress_mpa: float = 0.0
    safety_factor_z: float = 10.0
    tip_deflection_mm: float = 0.0
    stability_score: int = 100
    overall_verdict: Literal["stable", "caution", "unstable"] = "stable"
    recommended_wall_loops: int = 3
    recommended_infill_pct: int = 15
    recommended_material: str | None = None
    auto_remediations: list[str] = Field(default_factory=list)
    checks: list[PhysicsCheckItem] = Field(default_factory=list)


class DFMReport(BaseModel):
    watertight: bool
    manifold_repaired: bool = False
    triangle_count: int
    vertex_count: int
    bbox_mm: tuple[float, float, float]
    volume_cm3: float
    estimated_mass_g: float
    overhang_face_ratio: float = Field(ge=0.0, le=1.0)
    needs_supports: bool = False
    fits_build_plate: bool = True
    printer_id: str = "bambu_p1s"
    ams_slots_used: int = 1
    ams_units_configured: int = 1
    effective_multicolor_mode: str = "ams"
    joint_type_used: str = "none"
    coloring_method: str = "modular_solid"
    color_decisions: list[ColorDecision] = Field(default_factory=list)
    filament_recommendations: list[FilamentMatchRecommendation] = Field(default_factory=list)
    physics: PhysicsStabilityReport | None = None
    split_parts: list[SplitPartInfo] = Field(default_factory=list)
    checks: list[DFMCheckItem] = Field(default_factory=list)


class ExportBundle(BaseModel):
    threemf_path: str | None = None
    split_plates_3mf_path: str | None = None
    per_color_stls: dict[str, str] = Field(default_factory=dict)
    stl_path: str | None = None
    step_path: str | None = None
    preview_glb_path: str | None = None
    split_preview_glb_path: str | None = None
    turntable_preview_path: str | None = None
    cad_script_path: str | None = None
    gcode_3mf_path: str | None = None
    slicer_used: str | None = None
    estimated_print_minutes: float | None = None
    estimated_filament_g: float | None = None
    estimated_flush_g: float | None = None
    slicer_notes: str = ""


class ChatTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProjectState(BaseModel):
    project_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage: WorkflowStage = WorkflowStage.IDEATION
    spec: DesignSpec = Field(default_factory=DesignSpec)
    references: list[ReferenceAsset] = Field(default_factory=list)
    concepts: list[Concept2DAsset] = Field(default_factory=list)
    dfm: DFMReport | None = None
    exports: ExportBundle = Field(default_factory=ExportBundle)
    cad_code: str = ""
    history: list[ChatTurn] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
