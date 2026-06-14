"""
collect_renders.py
==================
Copy polylist_0.900000__front.png from each scene's KSR output folders
to a single collection directory, renaming each file as:

    <exp_name>_colmap.png
    <exp_name>_aps.png
    <exp_name>_aps_sampled.png

Usage
-----
    python KSR/collect_renders.py
"""

import glob
import os
import shutil
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MY_STORAGE  = "/Users/mchu/Documents/TUD/Thesis"

# Where all the renamed PNGs will land
COLLECT_DIR = f"{MY_STORAGE}/PolyGS/results/renders_collected"

SOURCE_PNG  = "polylist_0.900000__front.png"

SCENES = [
    {
        "exp_name": "church-cadeby",
        "colmap": f"{MY_STORAGE}/Pexels/church-cadeby/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/church-cadeby_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/church-cadeby_APS/*/planar_mesh.ply",
    },
    {
        "exp_name": "church-chesterfield",
        "colmap": f"{MY_STORAGE}/Pexels/church-chesterfield/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/church-chesterfield_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/church-chesterfield_APS/*/planar_mesh.ply",
    },
    {
        "exp_name": "killingbeck-cemetery",
        "colmap": f"{MY_STORAGE}/Pexels/killingbeck-cemetery/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/killingbeck-cemetery_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/killingbeck-cemetery_APS/*/planar_mesh.ply",
    },
    {
        "exp_name": "moskee-haarlem",
        "colmap": f"{MY_STORAGE}/Pexels/moskee-haarlem/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/moskee-haarlem_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/moskee-haarlem_APS/*/planar_mesh.ply",
    },
    {
        "exp_name": "tower-court",
        "colmap": f"{MY_STORAGE}/Pexels/tower-court/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/tower-court_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/tower-court_APS/*/planar_mesh.ply",
    },
    {
        "exp_name": "wotrubakirche",
        "colmap": f"{MY_STORAGE}/Pexels/wotrubakirche/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/wotrubakirche_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/wotrubakirche_APS/*/planar_mesh.ply",
    },
    {
        "exp_name": "elbphilharmonie",
        "colmap": f"{MY_STORAGE}/Pexels/elbphilharmonie/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/elbphilharmonie_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/elbphilharmonie_APS/*/planar_mesh.ply",
    },
    {
        "exp_name": "krasna-horka-castle",
        "colmap": f"{MY_STORAGE}/Pexels/krasna-horka-castle/fused_clipped.ply",
        "aps": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/krasna-horka-castle_APS/*/planar_mesh_for_KSR.ply",
        "aps_sampled": f"{MY_STORAGE}/PlanarSplatting/AdaptivePS/Pexels/krasna-horka-castle_APS/*/planar_mesh.ply",
    },
]


# ---------------------------------------------------------------------------
# Helper: copy one PNG if it exists
# ---------------------------------------------------------------------------
def collect(src_dir: str, dest_name: str, label: str) -> bool:
    src = os.path.join(src_dir, SOURCE_PNG)
    if not os.path.isfile(src):
        print(f"  [skip]  {label} — not found: {src}")
        return False
    dest = os.path.join(COLLECT_DIR, dest_name)
    shutil.copy2(src, dest)
    print(f"  [ok]    {label}  →  {dest_name}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(COLLECT_DIR, exist_ok=True)
    print(f"Collecting into: {COLLECT_DIR}\n")

    copied = 0
    missing = []

    for scene in SCENES:
        name = scene["exp_name"]
        print(f"[{name}]")

        # ---- colmap ---------------------------------------------------------
        colmap_ksr_dir = os.path.join(os.path.dirname(scene["colmap"]), "KSR")
        ok = collect(colmap_ksr_dir, f"{name}_colmap.png", f"{name}/colmap")
        if ok:
            copied += 1
        else:
            missing.append(f"{name}_colmap")

        # ---- aps ------------------------------------------------------------
        # The glob may match multiple experiment sub-dirs; take the first hit.
        aps_matches = sorted(glob.glob(scene["aps"]))
        if aps_matches:
            ksr_dir = os.path.join(os.path.dirname(aps_matches[0]), "KSR")
            ok = collect(ksr_dir, f"{name}_aps.png", f"{name}/aps")
            if ok:
                copied += 1
            else:
                missing.append(f"{name}_aps")
        else:
            print(f"  [skip]  {name}/aps — no glob match")
            missing.append(f"{name}_aps")

        # ---- aps_sampled ----------------------------------------------------
        aps_sampled_matches = sorted(glob.glob(scene["aps_sampled"]))
        if aps_sampled_matches:
            ksr_dir = os.path.join(os.path.dirname(aps_sampled_matches[0]), "KSR-sampled")
            ok = collect(ksr_dir, f"{name}_aps_sampled.png", f"{name}/aps_sampled")
            if ok:
                copied += 1
            else:
                missing.append(f"{name}_aps_sampled")
        else:
            print(f"  [skip]  {name}/aps_sampled — no glob match")
            missing.append(f"{name}_aps_sampled")

    # ---- Summary ------------------------------------------------------------
    total = len(SCENES) * 3
    print(f"\n{'='*60}")
    print(f"Copied {copied}/{total} images to {COLLECT_DIR}")
    if missing:
        print(f"Missing ({len(missing)}):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
