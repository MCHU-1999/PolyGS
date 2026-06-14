"""
batchrender_images.py
=====================
For every scene in SCENES, find all polylist_*.ply files produced by the KSR
batch run (in KSR/ and KSR-sampled/ subdirectories next to the input PLYs)
and render them with Easy3D using the scene's .view file.

Images are written to the SAME folder as the PLY files, named:
    <polylist_stem>__<view_stem>.png
e.g.  polylist_0.100000__front.png

Usage
-----
    python KSR/batchrender_images.py
"""

import glob
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Config — mirror batchrun.py
# ---------------------------------------------------------------------------
MY_STORAGE  = "/Users/mchu/Documents/TUD/Thesis"
POLYGS_ROOT = "/Users/mchu/Documents/TUD/Thesis/PolyGS"
BATCH_RENDER_PY = "/Users/mchu/Documents/TUD/Thesis/Easy3D/batch_render.py"

SCENES = [
    {
        "exp_name": "church-cadeby",
        "colmap": f"{MY_STORAGE}/Pexels/church-cadeby/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/church-cadeby_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/church-cadeby_APS/*/planar_mesh.ply",
        "view": f"{MY_STORAGE}/Pexels/church-cadeby/front.view",
    },
    {
        "exp_name": "church-chesterfield",
        "colmap": f"{MY_STORAGE}/Pexels/church-chesterfield/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/church-chesterfield_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/church-chesterfield_APS/*/planar_mesh.ply",
        "view": f"{MY_STORAGE}/Pexels/church-chesterfield/front.view",
    },
    {
        "exp_name": "killingbeck-cemetery",
        "colmap": f"{MY_STORAGE}/Pexels/killingbeck-cemetery/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/killingbeck-cemetery_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/killingbeck-cemetery_APS/*/planar_mesh.ply",
        "view": f"{MY_STORAGE}/Pexels/killingbeck-cemetery/front.view",
    },
    {
        "exp_name": "moskee-haarlem",
        "colmap": f"{MY_STORAGE}/Pexels/moskee-haarlem/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/moskee-haarlem_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/moskee-haarlem_APS/*/planar_mesh.ply",
        "view": f"{MY_STORAGE}/Pexels/moskee-haarlem/front.view",
    },
    {
        "exp_name": "tower-court",
        "colmap": f"{MY_STORAGE}/Pexels/tower-court/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/tower-court_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/tower-court_APS/*/planar_mesh.ply",
        "view": f"{MY_STORAGE}/Pexels/tower-court/front.view",
    },
    {
        "exp_name": "wotrubakirche",
        "colmap": f"{MY_STORAGE}/Pexels/wotrubakirche/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/wotrubakirche_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/wotrubakirche_APS/*/planar_mesh.ply",
        "view": f"{MY_STORAGE}/Pexels/wotrubakirche/front.view",
    },
    {
        "exp_name": "elbphilharmonie",
        "colmap": f"{MY_STORAGE}/Pexels/elbphilharmonie/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/elbphilharmonie_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/elbphilharmonie_APS/*/planar_mesh.ply",
        "view": f"{MY_STORAGE}/Pexels/elbphilharmonie/front.view",
    },
    {
        "exp_name": "krasna-horka-castle",
        "colmap": f"{MY_STORAGE}/Pexels/krasna-horka-castle/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/krasna-horka-castle_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/krasna-horka-castle_APS/*/planar_mesh.ply",
        "view": f"{MY_STORAGE}/Pexels/krasna-horka-castle/front.view",
    },
]

# The fixed set of polylist filenames produced by ksr_modified / ksr_buildings.
POLYLIST_NAMES = [
    "polylist_0.100000.ply",
    "polylist_0.300000.ply",
    "polylist_0.500000.ply",
    "polylist_0.700000.ply",
    "polylist_0.900000.ply",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_dir(output_dir: str, view_file: str, label: str) -> bool:
    """
    Find all polylist PLYs in output_dir and render them with batch_render.py.
    Images land in output_dir (same folder as the PLYs).
    Returns True on success.
    """
    plys = [
        os.path.join(output_dir, name)
        for name in POLYLIST_NAMES
        if os.path.isfile(os.path.join(output_dir, name))
    ]

    if not plys:
        print(f"  [{label}] No polylist PLYs found in {output_dir}, skipping.")
        return True  # not a failure — KSR may just not have run yet

    cmd = [
        sys.executable, BATCH_RENDER_PY,
        "--models", *plys,
        "--views",  view_file,
        "--output", output_dir,   # render into the same folder
        "--bg",     "white",
        "--edge-width", "2.0",
    ]

    print(f"\n{'='*70}")
    print(f"[{label}] Rendering {len(plys)} PLY(s) in {output_dir}")
    print(f"  view : {view_file}")
    print('='*70, flush=True)

    result = subprocess.run(cmd, cwd=POLYGS_ROOT)
    if result.returncode != 0:
        print(f"  [{label}] FAILED (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def render_file(ply_path: str, view_file: str, label: str) -> bool:
    """
    Render a single input PLY file with batch_render.py.
    Image lands in the same folder as the PLY.
    """
    if not os.path.isfile(ply_path):
        print(f"  [{label}] Input not found, skipping: {ply_path}")
        return True  # not a failure

    output_dir = os.path.dirname(ply_path)
    cmd = [
        sys.executable, BATCH_RENDER_PY,
        "--models", ply_path,
        "--views",  view_file,
        "--output", output_dir,
        "--bg",     "white",
        "--edge-width", "0",   # input point clouds / meshes — no forced edges
        "--point-size", "4.0",
    ]

    print(f"\n{'='*70}")
    print(f"[{label}] Rendering input: {os.path.basename(ply_path)}")
    print(f"  view : {view_file}")
    print('='*70, flush=True)

    result = subprocess.run(cmd, cwd=POLYGS_ROOT)
    if result.returncode != 0:
        print(f"  [{label}] FAILED (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def render_all():
    errors = []

    for scene in SCENES:
        name     = scene["exp_name"]
        view     = scene["view"]

        if not os.path.isfile(view):
            print(f"[{name}] View file not found, skipping all renders: {view}")
            errors.append(f"{name} (view missing)")
            continue

        # ---- COLMAP input --------------------------------------------------
        ok = render_file(scene["colmap"], view, label=f"{name}/colmap-input")
        if not ok:
            errors.append(f"{name}/colmap-input")

        # ---- COLMAP → KSR/ next to fused_clipped.ply ----------------------
        colmap_ksr_dir = os.path.join(os.path.dirname(scene["colmap"]), "KSR")
        ok = render_dir(colmap_ksr_dir, view, label=f"{name}/colmap")
        if not ok:
            errors.append(f"{name}/colmap")

        # ---- APS inputs + KSR/ ---------------------------------------------
        for aps_input in sorted(glob.glob(scene["aps"])):
            ok = render_file(aps_input, view, label=f"{name}/aps-input")
            if not ok:
                errors.append(f"{name}/aps-input ({aps_input})")
            ksr_dir = os.path.join(os.path.dirname(aps_input), "KSR")
            ok = render_dir(ksr_dir, view, label=f"{name}/aps")
            if not ok:
                errors.append(f"{name}/aps ({aps_input})")

        # ---- APS sampled inputs + KSR-sampled/ -----------------------------
        for aps_input in sorted(glob.glob(scene["aps_sampled"])):
            ok = render_file(aps_input, view, label=f"{name}/aps_sampled-input")
            if not ok:
                errors.append(f"{name}/aps_sampled-input ({aps_input})")
            ksr_dir = os.path.join(os.path.dirname(aps_input), "KSR-sampled")
            ok = render_dir(ksr_dir, view, label=f"{name}/aps_sampled")
            if not ok:
                errors.append(f"{name}/aps_sampled ({aps_input})")

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'='*70}")
    if errors:
        print(f"DONE — {len(errors)} error(s):")
        for e in errors:
            print(f"  FAILED: {e}")
        sys.exit(1)
    else:
        print("DONE — all renders completed successfully.")


if __name__ == "__main__":
    render_all()
