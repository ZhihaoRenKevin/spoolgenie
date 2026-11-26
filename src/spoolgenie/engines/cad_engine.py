"""Parametric CAD execution engine powered by build123d (Open CASCADE)."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trimesh

from spoolgenie.agent.cad_coder import (
    generate_template_cad_code,
    synthesize_cad_code,
    validate_cad_code_ast,
)
from spoolgenie.llm.client import UnifiedLLMClient
from spoolgenie.schemas import DesignSpec


@dataclass
class CADPartMesh:
    name: str
    slot: int
    mesh: trimesh.Trimesh


@dataclass
class CADBuildResult:
    parts: list[CADPartMesh]
    code_used: str
    step_path: str | None = None
    used_fallback: bool = False
    error_log: list[str] | None = None


def execute_build123d_script(
    code: str,
    step_output_path: str | Path | None = None,
    linear_deflection: float = 0.08,
    angular_deflection: float = 0.15,
) -> list[CADPartMesh]:
    """Validate and execute a build123d script, returning tessellated Trimesh parts + optional STEP."""
    validate_cad_code_ast(code)

    import build123d as b3d

    # Controlled execution namespace with only build123d symbols & math
    safe_builtins = {
        "range": range,
        "len": len,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "enumerate": enumerate,
        "zip": zip,
        "sum": sum,
        "__import__": _restricted_import,
    }
    ns: dict[str, Any] = {"__builtins__": safe_builtins, "math": math}
    for name in getattr(b3d, "__all__", dir(b3d)):
        if not name.startswith("_"):
            ns[name] = getattr(b3d, name)

    exec(compile(code, "<spoolgenie_cad>", "exec"), ns, ns)

    raw_parts = ns.get("PARTS")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise RuntimeError("CAD script did not define a non-empty PARTS list.")

    meshes: list[CADPartMesh] = []
    solids_for_step: list[Any] = []

    with tempfile.TemporaryDirectory(prefix="pf_cad_") as tmpdir:
        tmp = Path(tmpdir)
        for idx, item in enumerate(raw_parts):
            name = str(item.get("name") or f"part_{idx+1}")
            slot = max(1, min(16, int(item.get("slot", 1))))
            solid = item.get("solid")
            if solid is None:
                continue
            if isinstance(solid, (list, tuple)):
                solid = b3d.Compound(children=[s for s in solid if s is not None])
            solids_for_step.append(solid)
            stl_file = tmp / f"part_{idx}.stl"
            b3d.export_stl(
                solid,
                str(stl_file),
                tolerance=linear_deflection,
                angular_tolerance=angular_deflection,
            )
            loaded = trimesh.load(stl_file, force="mesh")
            if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
                meshes.append(CADPartMesh(name=name, slot=slot, mesh=loaded))

    if not meshes:
        raise RuntimeError("CAD script executed but produced zero valid triangle meshes.")

    if step_output_path and solids_for_step:
        out_step = Path(step_output_path)
        out_step.parent.mkdir(parents=True, exist_ok=True)
        compound = b3d.Compound(children=solids_for_step) if len(solids_for_step) > 1 else solids_for_step[0]
        b3d.export_step(compound, str(out_step))

    return meshes


def _restricted_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    root = name.split(".")[0]
    if root not in ("build123d", "math"):
        raise ImportError(f"Import of '{name}' is blocked inside SpoolGenie CAD sandbox.")
    return __import__(name, globals, locals, fromlist, level)


def build_parametric_model(
    spec: DesignSpec,
    step_output_path: str | Path | None = None,
    llm: UnifiedLLMClient | None = None,
    custom_code: str | None = None,
    max_retries: int = 3,
    kb_context: str = "",
) -> CADBuildResult:
    """Build multi-part parametric CAD geometry with automatic LLM self-repair and template fallback."""
    errors: list[str] = []
    code = custom_code or synthesize_cad_code(spec, llm=llm, kb_context=kb_context)

    for attempt in range(max_retries + 1):
        try:
            parts = execute_build123d_script(code, step_output_path=step_output_path)
            return CADBuildResult(
                parts=parts,
                code_used=code,
                step_path=str(step_output_path) if step_output_path else None,
                used_fallback=False,
                error_log=errors,
            )
        except Exception as exc:
            errors.append(f"Attempt {attempt+1}: {exc}")
            if custom_code or attempt >= max_retries or not (llm and llm.available):
                break
            code = synthesize_cad_code(
                spec,
                llm=llm,
                previous_error=str(exc),
                previous_code=code,
                kb_context=kb_context,
            )

    # Guaranteed fallback to parametric template
    fallback_code = generate_template_cad_code(spec)
    parts = execute_build123d_script(fallback_code, step_output_path=step_output_path)
    return CADBuildResult(
        parts=parts,
        code_used=fallback_code,
        step_path=str(step_output_path) if step_output_path else None,
        used_fallback=True,
        error_log=errors,
    )
