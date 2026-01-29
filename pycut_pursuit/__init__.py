### INIT
import os
import sys

# locate wrappers and bin relative to this package (repo layout)
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_this_dir, ".."))
_wrappers_dir = os.path.join(_repo_root, "pycut_pursuit", "python", "wrappers")
_bin_dir = os.path.join(_repo_root, "pycut_pursuit", "python", "bin")

# add them to sys.path (front) if they exist
if os.path.isdir(_wrappers_dir) and _wrappers_dir not in sys.path:
    sys.path.insert(0, _wrappers_dir)
if os.path.isdir(_bin_dir) and _bin_dir not in sys.path:
    sys.path.insert(0, _bin_dir)

try:
    from cp_d0_dist import cp_d0_dist
    from cp_d1_lsx import cp_d1_lsx
    from cp_d1_ql1b import cp_d1_ql1b
    from cp_prox_tv import cp_prox_tv
except Exception as e:
    raise ImportError(
        "pycut_pursuit: failed to import wrappers. Ensure pycut_pursuit/python/wrappers "
        "and pycut_pursuit/python/bin exist or set PYTHONPATH appropriately."
    ) from e
# ...existing code...