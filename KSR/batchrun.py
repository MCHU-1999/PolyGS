MY_STORAGE = "/Users/mchu/Documents/TUD/Thesis"
POLYGS_ROOT = "/Users/mchu/Documents/TUD/Thesis/PolyGS"
SCENES = [
    # Pexels Datasets
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

# ---------------------------------------------------------------------------
import glob
import os
import subprocess
import sys

KSR_BUILDINGS = os.path.join(POLYGS_ROOT, "build/KSR/ksr_buildings")
KSR_MODIFIED  = os.path.join(POLYGS_ROOT, "build/KSR/ksr_modified")


def run(cmd: list[str], label: str) -> bool:
    """Run a subprocess, stream output, return True on success."""
    print(f"\n{'='*70}")
    print(f"[{label}] {' '.join(cmd)}")
    print('='*70, flush=True)
    result = subprocess.run(cmd, cwd=POLYGS_ROOT)
    if result.returncode != 0:
        print(f"[{label}] FAILED (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def run_all():
    errors = []

    for scene in SCENES:
        name = scene["exp_name"]

        # ---- COLMAP → ksr_buildings ----------------------------------------
        colmap_input = scene["colmap"]
        if os.path.isfile(colmap_input):
            output_dir = os.path.join(os.path.dirname(colmap_input), "KSR")
            ok = run(
                [KSR_BUILDINGS, "-i", colmap_input, "-o", output_dir],
                label=f"{name}/colmap",
            )
            if not ok:
                errors.append(f"{name}/colmap")
        else:
            print(f"[{name}/colmap] Input not found, skipping: {colmap_input}")

        # ---- APS → ksr_modified (glob over experiment sub-directories) ------
        aps_matches = sorted(glob.glob(scene["aps"]))
        if not aps_matches:
            print(f"[{name}/aps] No matches for glob, skipping: {scene['aps']}")
        for aps_input in aps_matches:
            output_dir = os.path.join(os.path.dirname(aps_input), "KSR")
            ok = run(
                [KSR_MODIFIED, "-i", aps_input, "-o", output_dir],
                label=f"{name}/aps",
            )
            if not ok:
                errors.append(f"{name}/aps ({aps_input})")

        # ---- APS sampled → ksr_modified (glob over experiment sub-directories)
        aps_sampled_matches = sorted(glob.glob(scene["aps_sampled"]))
        if not aps_sampled_matches:
            print(f"[{name}/aps_sampled] No matches for glob, skipping: {scene['aps_sampled']}")
        for aps_input in aps_sampled_matches:
            output_dir = os.path.join(os.path.dirname(aps_input), "KSR-sampled")
            ok = run(
                [KSR_MODIFIED, "-i", aps_input, "-o", output_dir],
                label=f"{name}/aps_sampled",
            )
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
        print("DONE — all scenes completed successfully.")


if __name__ == "__main__":
    run_all()