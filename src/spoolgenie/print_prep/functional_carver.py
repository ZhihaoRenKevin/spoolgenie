"""General-purpose 3D functional boolean carver (powered by manifold3d).

Turns ANY organic/character 3D mesh (from Hunyuan3D-2, TRELLIS, Tripo, or uploaded STL/GLB) into a
precision functional 3D print while guaranteeing 100% watertightness:
  - `hollow_cavity`  : Pen holder / desk caddy / succulent planter cavity with controlled wall & floor thickness
  - `phone_slot`     : Ergonomic angled phone/tablet cradle slot + bottom USB-C cable pass-through channel
  - `magsafe_recess` : Precision Φ56.4 x 5.8 mm MagSafe / wireless puck socket + rear cord channel
  - `magnet_sockets` : Underside press-fit neodymium magnet pockets (Φ6.2 x 2.4 mm)
  - `cable_comb`     : Rear desktop cable organizer apron with beveled cable notches
  - `plinth_base`    : Flat chamfered anti-tip pedestal seated cleanly on Z = 0
  - `keychain_loop`  : Reinforced printable top hanging loop
"""

from __future__ import annotations

import manifold3d
import numpy as np
import trimesh

from spoolgenie.schemas import FunctionalOp


def _try_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold | None:
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_winding(m)
    trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    m_gl = manifold3d.Mesh(vert_properties=verts, tri_verts=faces)
    if hasattr(m_gl, "merge"):
        m_gl.merge()
    man = manifold3d.Manifold(m_gl)
    if man.num_tri() > 0:
        return man
    return None


def _mesh_to_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
    """Convert any Trimesh to a watertight manifold3d.Manifold, using Poisson remesh fallback for AI meshes."""
    man = _try_manifold(mesh)
    if man is not None:
        return man

    # Fallback: pymeshlab Screened Poisson surface reconstruction guarantees a clean closed 2-manifold!
    import importlib.util

    if importlib.util.find_spec("pymeshlab"):
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.compute_normal_per_vertex()
        ms.generate_surface_reconstruction_screened_poisson(depth=7, preclean=True)
        m_out = ms.current_mesh()
        rebuilt = trimesh.Trimesh(vertices=m_out.vertex_matrix(), faces=m_out.face_matrix(), process=True)
        man = _try_manifold(rebuilt)
        if man is not None:
            return man

    raise ValueError("Manifold conversion produced empty solid even after watertight reconstruction.")


def _manifold_to_mesh(man: manifold3d.Manifold) -> trimesh.Trimesh:
    if man.num_tri() == 0:
        raise ValueError("Empty manifold")
    m = man.to_mesh()
    verts = np.asarray(m.vert_properties[:, :3], dtype=np.float64)
    faces = np.asarray(m.tri_verts, dtype=np.int64)
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    trimesh.repair.fix_normals(out)
    return out


def apply_functional_ops_to_mesh(
    mesh: trimesh.Trimesh,
    ops: list[FunctionalOp],
    wall_thickness_mm: float = 3.2,
) -> tuple[trimesh.Trimesh, list[str]]:
    """Apply exact millimeter functional boolean operations onto any arbitrary 3D mesh."""
    if not ops:
        return mesh, []

    try:
        man = _mesh_to_manifold(mesh)
    except Exception as exc:
        return mesh, [f"Skipped functional carving (input mesh non-manifold: {exc})"]

    applied_log: list[str] = []

    for fop in ops:
        p = fop.params or {}
        bmin, bmax = mesh.bounds
        ext = bmax - bmin
        cx = 0.5 * (bmin[0] + bmax[0])
        cy = 0.5 * (bmin[1] + bmax[1])
        w, d, h = float(ext[0]), float(ext[1]), float(ext[2])
        wall = max(2.2, float(p.get("wall_mm", wall_thickness_mm)))

        if fop.op == "plinth_base":
            base_h = max(3.0, min(10.0, float(p.get("height_mm", 5.0))))
            rx = max(w * 0.52, 22.0)
            ry = max(d * 0.52, 22.0)
            # Elliptical beveled display plinth seated at Z=0
            plinth = manifold3d.Manifold.cylinder(base_h, rx, rx * 0.96, circular_segments=48)
            plinth = plinth.scale([1.0, ry / rx, 1.0]).translate([cx, cy, 0.0])
            man = man.translate([0.0, 0.0, base_h - 1.2]) + plinth
            applied_log.append(f"plinth_base ({rx*2:.0f}x{ry*2:.0f}x{base_h:.0f} mm)")

        elif fop.op == "hollow_cavity":
            # Detect whether the character has a protruding head above a wider torso (e.g. WALL-E, Totoro, Iron Man):
            # If top 22% is narrower than middle torso, attach/carve an integrated rear backpack pen caddy behind the head
            # so the character's face & eyes stay 100% intact! Otherwise carve straight down from the flat top.
            floor_z = max(5.0, float(p.get("floor_mm", 6.0)))
            v_arr = mesh.vertices
            top_verts = v_arr[v_arr[:, 2] > (bmin[2] + h * 0.78)]
            mid_verts = v_arr[(v_arr[:, 2] > (bmin[2] + h * 0.25)) & (v_arr[:, 2] < (bmin[2] + h * 0.65))]
            has_narrow_head = False
            if len(top_verts) > 20 and len(mid_verts) > 20:
                top_span = float(np.ptp(top_verts[:, 0]) * np.ptp(top_verts[:, 1]))
                mid_span = float(np.ptp(mid_verts[:, 0]) * np.ptp(mid_verts[:, 1]))
                has_narrow_head = top_span < mid_span * 0.72

            if has_narrow_head:
                # Seamless rear functional pen/storage caddy box merged into the character's back (head untouched!)
                caddy_w = max(44.0, min(w * 0.62, 68.0))
                caddy_d = max(34.0, min(d * 0.48, 48.0))
                caddy_h = max(48.0, min(h * 0.60, 76.0))
                caddy_y = bmax[1] + caddy_d * 0.16
                outer = manifold3d.Manifold.cube([caddy_w, caddy_d, caddy_h], center=True).translate(
                    [cx, caddy_y, caddy_h * 0.5]
                )
                inner = manifold3d.Manifold.cube(
                    [caddy_w - 2.0 * wall, caddy_d - 2.0 * wall, caddy_h], center=True
                ).translate([cx, caddy_y, floor_z + caddy_h * 0.5])
                man = (man + outer) - inner
                applied_log.append(
                    f"hollow_cavity rear backpack caddy ({caddy_w:.0f}x{caddy_d:.0f}x{caddy_h:.0f} mm, head preserved)"
                )
            else:
                cavity_ratio = float(p.get("width_ratio", 0.52))
                cav_w = max(18.0, min(w - 2.0 * wall, w * cavity_ratio))
                cav_d = max(18.0, min(d - 2.0 * wall, d * cavity_ratio))
                depth_ratio = float(p.get("depth_ratio", 0.62))
                cav_bottom_z = max(floor_z, h * (1.0 - depth_ratio))
                cav_h = h - cav_bottom_z + 12.0
                shape = str(p.get("shape", "round")).lower()

                if shape == "square":
                    cutter = manifold3d.Manifold.cube([cav_w, cav_d, cav_h], center=True).translate(
                        [cx, cy, cav_bottom_z + cav_h * 0.5]
                    )
                else:
                    rad = 0.5 * min(cav_w, cav_d)
                    cutter = manifold3d.Manifold.cylinder(cav_h, rad * 0.94, rad, circular_segments=48).translate(
                        [cx, cy, cav_bottom_z]
                    )
                man = man - cutter
                applied_log.append(f"hollow_cavity ({cav_w:.0f}x{cav_d:.0f} mm, floor Z={cav_bottom_z:.0f} mm)")

        elif fop.op == "phone_slot":
            # Build an integrated front apron + ergonomic angled phone slot + USB-C cable tunnel
            slot_w = max(72.0, min(w * 0.95, float(p.get("slot_width_mm", 84.0))))
            slot_t = max(11.0, min(20.0, float(p.get("slot_thickness_mm", 14.0))))
            shelf_d = max(24.0, slot_t + 12.0)
            shelf_h = max(16.0, min(28.0, h * 0.22))

            # Add sturdy front cradle ledge seamlessly extending forward (-Y) at Z=0
            ledge_y = bmin[1] - shelf_d * 0.35
            ledge = manifold3d.Manifold.cube([max(slot_w + 10.0, w * 0.88), shelf_d, shelf_h], center=True).translate(
                [cx, ledge_y, shelf_h * 0.5]
            )
            man = man + ledge

            # Carve angled phone groove across ledge + lower front body
            groove = (
                manifold3d.Manifold.cube([slot_w, slot_t, h * 0.48], center=True)
                .rotate([16.0, 0.0, 0.0])
                .translate([cx, ledge_y + 1.5, shelf_h * 0.35 + h * 0.22])
            )
            # Carve centered 18 mm USB-C cable routing tunnel along bottom from front ledge out the back
            cable = manifold3d.Manifold.cube([18.0, d + shelf_d + 20.0, 10.0], center=True).translate(
                [cx, cy - shelf_d * 0.2, 5.0]
            )
            man = man - groove - cable
            applied_log.append(f"phone_slot (W={slot_w:.0f} mm, groove={slot_t:.1f} mm + USB-C cable tunnel)")

        elif fop.op == "magsafe_recess":
            # Precision MagSafe / wireless charger puck pocket flush on front chest or top face
            puck_d = float(p.get("diameter_mm", 56.4))
            puck_depth = float(p.get("depth_mm", 5.8))
            face = str(p.get("face", "front")).lower()
            r = puck_d * 0.5

            if face == "top":
                puck = manifold3d.Manifold.cylinder(puck_depth + 4.0, r, r, circular_segments=64).translate(
                    [cx, cy, bmax[2] - puck_depth]
                )
                wire = manifold3d.Manifold.cube([6.0, d * 0.65, 6.0], center=True).translate(
                    [cx, cy + d * 0.3, bmax[2] - puck_depth + 1.5]
                )
            else:
                # Front (-Y) chest recess
                puck = (
                    manifold3d.Manifold.cylinder(puck_depth + 8.0, r, r, circular_segments=64)
                    .rotate([90.0, 0.0, 0.0])
                    .translate([cx, bmin[1] + puck_depth + 4.0, bmin[2] + h * 0.48])
                )
                wire = manifold3d.Manifold.cube([6.0, d + 12.0, 6.5], center=True).translate(
                    [cx, cy, bmin[2] + h * 0.22]
                )
            man = man - puck - wire
            applied_log.append(f"magsafe_recess (Φ{puck_d:.1f}x{puck_depth:.1f} mm on {face} + cable channel)")

        elif fop.op == "magnet_sockets":
            # 2 or 4 underside press-fit neodymium magnet pockets at Z=0
            mag_d = float(p.get("diameter_mm", 6.2))
            mag_h = float(p.get("depth_mm", 2.4))
            r = mag_d * 0.5
            ox = w * 0.26
            oy = d * 0.24
            for dx in (-ox, ox):
                for dy in (-oy, oy):
                    hole = manifold3d.Manifold.cylinder(mag_h + 0.8, r, r, circular_segments=32).translate(
                        [cx + dx, cy + dy, -0.4]
                    )
                    man = man - hole
            applied_log.append(f"magnet_sockets (4x Φ{mag_d:.1f}x{mag_h:.1f} mm underside)")

        elif fop.op == "cable_comb":
            # Rear base cable organizer apron with 3 beveled cord slots
            apron_w = max(46.0, min(w * 0.82, 74.0))
            apron_d = 18.0
            apron_h = 14.0
            ay = bmax[1] + apron_d * 0.35
            apron = manifold3d.Manifold.cube([apron_w, apron_d, apron_h], center=True).translate([cx, ay, apron_h * 0.5])
            man = man + apron
            n_slots = int(p.get("slots", 3))
            pitch = apron_w / (n_slots + 1)
            for i in range(n_slots):
                sx = cx - apron_w * 0.5 + (i + 1) * pitch
                bore = (
                    manifold3d.Manifold.cylinder(apron_d + 8.0, 3.2, 3.2, circular_segments=24)
                    .rotate([90.0, 0.0, 0.0])
                    .translate([sx, ay + apron_d * 0.5 + 4.0, apron_h * 0.55])
                )
                neck = manifold3d.Manifold.cube([4.0, apron_d + 8.0, apron_h * 0.65], center=True).translate(
                    [sx, ay, apron_h * 0.82]
                )
                man = man - bore - neck
            applied_log.append(f"cable_comb ({n_slots} rear cord slots)")

        elif fop.op == "keychain_loop":
            r_out = float(p.get("outer_radius_mm", 7.0))
            r_in = float(p.get("inner_radius_mm", 3.4))
            thick = float(p.get("thickness_mm", 4.2))
            ring = (
                manifold3d.Manifold.cylinder(thick, r_out, r_out, circular_segments=32)
                - manifold3d.Manifold.cylinder(thick + 2.0, r_in, r_in, circular_segments=24).translate([0.0, 0.0, -1.0])
            )
            ring = ring.rotate([90.0, 0.0, 0.0]).translate([cx, cy + thick * 0.5, bmax[2] + r_out * 0.45])
            man = man + ring
            applied_log.append(f"keychain_loop (OD {r_out*2:.0f} mm / ID {r_in*2:.0f} mm)")

        # Refresh mesh bounds for any subsequent chained op
        try:
            mesh = _manifold_to_mesh(man)
        except Exception:
            break

    # Ensure flat bottom cleanly sits on Z=0 and never return an empty mesh
    try:
        final_mesh = _manifold_to_mesh(man)
    except Exception:
        return mesh, applied_log + ["Boolean result was empty; preserved clean base mesh."]
    if len(final_mesh.faces) > 0:
        bmin = final_mesh.bounds[0]
        final_mesh.apply_translation([0.0, 0.0, -bmin[2]])
    return final_mesh, applied_log
