"""Cross-platform headless Bambu Studio / OrcaSlicer CLI runner with physics-based fallback estimates."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from spoolgenie.config import SpoolGenieSettings, get_settings
from spoolgenie.doctor import find_slicer_binaries
from spoolgenie.schemas import DesignSpec, DFMReport, MulticolorMode


@dataclass
class SliceOutcome:
    sliced: bool
    slicer_used: str | None
    gcode_3mf_path: str | None
    estimated_print_minutes: float
    estimated_filament_g: float
    estimated_flush_g: float
    notes: str


def estimate_print_metrics(dfm: DFMReport, spec: DesignSpec) -> tuple[float, float, float]:
    """Physics-based heuristic estimate when Bambu Studio / OrcaSlicer CLI isn't installed locally."""
    filament_g = max(2.0, dfm.estimated_mass_g)
    # Typical Bambu CoreXY throughput ~ 14-18 g/hr (~0.26 g/min) for 0.2mm PLA + overhead
    base_minutes = 12.0 + filament_g / 0.26

    # AMS multicolor swap flush penalty ONLY applies in AMS mode (0 poop in No-AMS split mode!)
    if dfm.effective_multicolor_mode == MulticolorMode.AMS.value and dfm.ams_slots_used > 1:
        est_layers = max(10.0, dfm.bbox_mm[2] / 0.2)
        est_swaps = min(est_layers * 0.45 * (dfm.ams_slots_used - 1), 180.0)
        flush_g = round(est_swaps * 0.42, 1)
        swap_minutes = est_swaps * 1.15
    else:
        flush_g = 0.0
        swap_minutes = 0.0

    support_mult = 1.15 if dfm.needs_supports else 1.0
    total_minutes = round((base_minutes + swap_minutes) * support_mult, 1)
    return total_minutes, round(filament_g + flush_g, 1), flush_g


def run_slicer_or_estimate(
    threemf_path: str | Path,
    output_dir: str | Path,
    dfm: DFMReport,
    spec: DesignSpec,
    settings: SpoolGenieSettings | None = None,
) -> SliceOutcome:
    """Attempt headless Bambu Studio / OrcaSlicer CLI slicing; return clean estimate if slicer is not installed."""
    settings = settings or get_settings()
    slicers = find_slicer_binaries(settings.slicer_path)
    slicer_name = None
    slicer_bin = None
    if slicers.get("orca_slicer"):
        slicer_name, slicer_bin = "OrcaSlicer", slicers["orca_slicer"]
    elif slicers.get("bambu_studio"):
        slicer_name, slicer_bin = "Bambu Studio", slicers["bambu_studio"]

    est_min, est_g, est_flush = estimate_print_metrics(dfm, spec)

    if not slicer_bin:
        return SliceOutcome(
            sliced=False,
            slicer_used=None,
            gcode_3mf_path=None,
            estimated_print_minutes=est_min,
            estimated_filament_g=est_g,
            estimated_flush_g=est_flush,
            notes=(
                "Bambu Studio / OrcaSlicer not detected locally — generated ready-to-slice multicolor .3mf "
                "with physics-based time/mass estimates. Install Bambu Studio or OrcaSlicer for automatic .gcode.3mf output."
            ),
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(threemf_path).stem}.gcode.3mf"

    cmd = [
        slicer_bin,
        "--arrange",
        "1",
        "--slice",
        "0",
        "--export-3mf",
        str(out_file),
        str(threemf_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        combined_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode == 0 and out_file.exists():
            # Try parsing slicer-reported duration/filament if logged
            m_g = re.search(r"total filament used.*?(\d+(?:\.\d+)?)\s*g", combined_out, flags=re.I)
            if m_g:
                est_g = float(m_g.group(1))
            return SliceOutcome(
                sliced=True,
                slicer_used=slicer_name,
                gcode_3mf_path=str(out_file.resolve()),
                estimated_print_minutes=est_min,
                estimated_filament_g=est_g,
                estimated_flush_g=est_flush,
                notes=f"Sliced headlessly via {slicer_name} CLI.",
            )
        return SliceOutcome(
            sliced=False,
            slicer_used=slicer_name,
            gcode_3mf_path=None,
            estimated_print_minutes=est_min,
            estimated_filament_g=est_g,
            estimated_flush_g=est_flush,
            notes=f"{slicer_name} CLI exited with code {proc.returncode}; open the exported .3mf directly in {slicer_name}.",
        )
    except Exception as exc:
        return SliceOutcome(
            sliced=False,
            slicer_used=slicer_name,
            gcode_3mf_path=None,
            estimated_print_minutes=est_min,
            estimated_filament_g=est_g,
            estimated_flush_g=est_flush,
            notes=f"{slicer_name} CLI failed ({exc}); open the exported .3mf directly in {slicer_name}.",
        )
