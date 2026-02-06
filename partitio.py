# python partition.py --ply example_pointcloud/dtu_scan24.ply
import argparse
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from pycut_pursuit import cp_d0_dist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plyfile import PlyData, PlyElement

# Features list
FEATURE_KEYS = [
    'x', 'y', 'z',                      # Just XYZ
    'nx', 'ny', 'nz',                   # Normal vectors
    'f_dc_0', 'f_dc_1', 'f_dc_2',       # Color
    'scale_0', 'scale_1', 'scale_2',    # Scale
    'rot_0', 'rot_1', 'rot_2', 'rot_3', # Rotation
    'opacity'
]
# Weights
WEIGHTS = np.array([
    10, 10, 10,
    30, 30, 30,
    1, 1, 1,
    0, 0, 0,
    0, 0, 0, 0,
    0
], dtype=np.float32)

def load_ply(ply_file, feature_keys, verbose: bool):
    # Load PLY file
    pc = PlyData.read(ply_file)
    elements = [e.name for e in pc.elements]
    if 'vertex' not in elements:
        raise RuntimeError(f"PLY has no 'vertex' element. Found elements: {elements}")
    data = pc['vertex'].data
    names = data.dtype.names
    for f in feature_keys:
        if f not in names:
            raise ValueError("The points in PLY file do not contain the necessary features")
    selected_features = np.vstack([data[k] for k in feature_keys]).T.astype(np.float32)
    # Success
    print(f'Loaded {selected_features.shape[0]} points, using features: {feature_keys}')

    if verbose:
        # Print first 20 points and their feature values and exit
        m = min(20, selected_features.shape[0])
        print(f"First {m} points and their features (columns = {feature_keys}):")
        np.set_printoptions(precision=6, suppress=True)
        for i in range(m):
            row = ", ".join(f"{feature_keys[j]}={selected_features[i, j]:.6g}" for j in range(selected_features.shape[1]))
            print(f"{i}: {row}")
    
    return selected_features

def build_forward_star(n, neigh_lists):
    # neigh_lists: list of neighbor lists for each node (directed)
    ptr = np.zeros(n + 1, dtype=np.uint32)
    lengths = [len(l) for l in neigh_lists]
    ptr[1:] = np.cumsum(lengths).astype(np.uint32)
    targets = np.empty(ptr[-1], dtype=np.uint32)
    idx = 0
    for l in neigh_lists:
        for v in l:
            targets[idx] = np.uint32(v)
            idx += 1
    return ptr, targets

# remove rows with non-finite coordinates (first 3 columns)
def remove_nonfinite_coords(X):
    """Remove rows from X that have non-finite coordinates.

    Parameters
    - X: numpy array (n x D)
    - coords: optional (n x 3) array to check instead of X[:, :3]
    - verbose: print examples of bad rows
    - max_examples: number of bad rows to print when verbose=True

    Returns
    - X_filtered: filtered numpy array
    - finite_mask: boolean mask of kept rows
    """

    coords = X[:, :3]
    finite_mask = np.isfinite(coords).all(axis=1)
    if finite_mask.all():
        return X, finite_mask

    n_bad = int((~finite_mask).sum())
    print(f'Removing {n_bad} points with non-finite coordinates')

    X_filtered = X[finite_mask]
    if X_filtered.size == 0:
        raise RuntimeError('No finite points remain after filtering')
    return X_filtered

def keep_largest_connected_component(neigh, dists, X, coords):
    """Keep only the largest connected component from a directed neighbor list.

    Parameters
    - neigh: list of neighbor lists for each node
    - dists: list/array of corresponding distances per neighbor
    - X: feature matrix (n x D)
    - coords: coordinate matrix (n x 3)

    Returns: (neigh, dists, X, coords, n)
    """

    from collections import deque
    n = X.shape[0]
    # build undirected adjacency
    adj = [set() for _ in range(n)]
    for i, lst in enumerate(neigh):
        for j in lst:
            adj[i].add(j)
            adj[j].add(i)

    visited = np.zeros(n, dtype=bool)
    best_comp = []
    for start in range(n):
        if visited[start]:
            continue
        q = deque([start])
        visited[start] = True
        comp = [start]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
                    comp.append(v)
        if len(comp) > len(best_comp):
            best_comp = comp

    best_comp = np.array(sorted(best_comp), dtype=np.int32)
    if best_comp.size != n:
        print(f'Keeping largest connected component: {best_comp.size} / {n} nodes')
        # create mapping old->new
        mapping = np.full(n, -1, dtype=np.int32)
        mapping[best_comp] = np.arange(best_comp.size, dtype=np.int32)
        # filter neigh and corresponding dists
        new_neigh = []
        new_dists = []
        for old_i in best_comp:
            lst = []
            ld = []
            for kk, j in enumerate(neigh[old_i]):
                mj = mapping[j]
                if mj != -1:
                    lst.append(int(mj))
                    ld.append(float(dists[old_i][kk]))
            new_neigh.append(lst)
            new_dists.append(np.array(ld, dtype=np.float32))
        X_new = X[best_comp]
        coords_new = X_new[:, :3]
        n_new = X_new.shape[0]
        return new_neigh, new_dists, X_new, coords_new, n_new
    else:
        print('Graph already fully connected; nothing to do.')
        return neigh, dists, X, coords, n
    
def flip_coords(data):
    """
    Flip CV coordinates (Y-down, Z-forward) to OpenGL (Y-up, Z-back).
    Works on point clouds (N, 3) or transformation matrices (4, 4).
    """
    # Ensure data is a numpy array
    data = np.asanyarray(data)
    
    # For Point Clouds: [x, y, z] -> [x, -y, -z]
    # Using slice notation for in-place speed
    data[:, 1:3] *= -1
    return data

def visualize_open3d(coords, first_edge, target, labels=None):
    try:
        import open3d as o3d
    except Exception as e:
        print('Open3D not available:', e); return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    if labels is not None:
        nlabels = int(labels.max()) + 1
        rng = np.random.RandomState(42)
        palette = (rng.randint(0, 256, size=(nlabels, 3)) / 255.0)
        pcd.colors = o3d.utility.Vector3dVector(palette[labels])
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.7, 0.7, 0.7]), (coords.shape[0], 1)))

    lines = []
    for i in range(coords.shape[0]):
        for e in range(first_edge[i], first_edge[i + 1]):
            j = int(target[e])
            if 0 <= j < coords.shape[0]:
                lines.append([i, j])
    if len(lines) == 0:
        print('No edges to display.')
        o3d.visualization.draw_geometries([pcd]); return
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(coords),
                                    lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32)))
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.8, 0.8, 0.8]), (len(lines), 1)))
    o3d.visualization.draw_geometries([pcd, line_set], window_name='Graph', width=1024, height=768)

def write_colored_ply(X, super_index, out_path, feature_names, rng_seed=42):
    """Write a PLY that preserves all columns in `X` (using `feature_names`) and
    appends `red/green/blue` and `f_dc_0/1/2` fields.

    - If `feature_names` is None the module-level `FEATURE_KEYS` is used.
    - f_dc_* are taken from X when present (columns named 'f_dc_0/1/2'), otherwise synthesized.
    - RGB is derived from `super_index` palette mapping.
    """

    labels = np.asarray(super_index).astype(np.int64)
    if labels.size == 0:
        raise ValueError('super_index is empty')
    num_labels = int(labels.max()) + 1

    rng = np.random.RandomState(rng_seed)
    palette = rng.randint(0, 256, size=(num_labels, 3)).astype(np.uint8)
    colors = palette[labels]

    coords = np.ascontiguousarray(X[:, :3].astype(np.float32))
    n = coords.shape[0]

    # Always synthesize f_dc_* from the palette and overwrite any existing values
    fdc = (colors.astype(np.float32) / 255.0) * 4.0 - 2.0
    fdc = np.clip(fdc, -2.0, 2.0)

    # Build dtype: include all original feature names (float32), then add RGB (u1) and f_dc (f4)
    descr = []
    for name in feature_names:
        descr.append((name, 'f4'))
    # add RGB
    for name in ("red", "green", "blue"):
        if name not in feature_names:
            descr.append((name, 'u1'))

    dtype = np.dtype(descr)
    vertex = np.empty(n, dtype=dtype)

    # Fill original features (if present in X)
    for j, name in enumerate(feature_names):
        vertex[name] = X[:, j].astype(np.float32)

    # Fill f_dc fields (either from X or synthesized)
    vertex['f_dc_0'] = fdc[:, 0]
    vertex['f_dc_1'] = fdc[:, 1]
    vertex['f_dc_2'] = fdc[:, 2]

    # Fill RGB
    vertex['red'] = colors[:, 0].astype(np.uint8)
    vertex['green'] = colors[:, 1].astype(np.uint8)
    vertex['blue'] = colors[:, 2].astype(np.uint8)

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(out_path)
    print(f'Wrote colored PLY to: {out_path}')


# =============================================================================================
# Main function
# =============================================================================================
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Partition a PLY pointcloud with cp_d0_dist')
    parser.add_argument('--ply', required=True, help='Path to input PLY file')
    parser.add_argument('--out', help='Path to output PLY file')
    parser.add_argument('--k', type=int, default=20, help='k for k-NN graph')
    parser.add_argument('--min-comp', type=float, default=10.0, help='min component weight (points)')
    parser.add_argument('--max-it', type=int, default=30, help='cp_d0_dist max iterations')
    parser.add_argument('--keep-largest', type=bool, default=True, help='Keep only the largest connected component of the k-NN graph')
    parser.add_argument('--verbose', action='store_true', help='Whether to print the point cloud feature values or not')
    args = parser.parse_args()

    # Load PLY file
    ply_file = args.ply
    selected_features = load_ply(ply_file, FEATURE_KEYS, args.verbose)
    X = remove_nonfinite_coords(selected_features)
    n = X.shape[0]
    D = selected_features.shape[1]

    # k-NN graph on coordinates
    k = args.k
    coords = X[:,:3]
    tree = cKDTree(coords)
    dists, inds = tree.query(coords, k=k+1)
    dists = dists[:, 1:]
    inds = inds[:, 1:]
    neigh = [list(map(int, inds[i])) for i in range(n)]

    if args.keep_largest:
        neigh, dists, X, coords, n = keep_largest_connected_component(neigh, dists, X, coords)

    x = np.asfortranarray(X.T.astype(np.float32))
    first_edge, target = build_forward_star(n, neigh)

    # Edge weights: exponential kernel
    eps = 1e-12
    flat_dists = np.concatenate([dists[i] for i in range(n)]).astype(np.float32)
    mean_dist = flat_dists.mean()
    edge_weights = np.exp(-flat_dists / (mean_dist + eps)).astype(np.float32)

    # Vertex weights: atomic points -> 1
    vert_weights = np.ones(n, dtype=np.float32)

    # Run partition
    start_time = time.time()
    super_index, x_c, cluster, edges, times = cp_d0_dist(
        D,
        x,
        first_edge.astype(np.uint32),
        target.astype(np.uint32),
        edge_weights=edge_weights,
        vert_weights=None,
        coor_weights=WEIGHTS,
        min_comp_weight=args.min_comp,
        cp_dif_tol=1e-2,
        cp_it_max=args.max_it,
        split_damp_ratio=0.7,
        verbose=args.verbose,
        max_num_threads=0,
        balance_parallel_split=True,
        compute_Time=True,
        compute_List=True,
        compute_Graph=True
    )
    exec_time = time.time() - start_time

    print('n nodes:', n)
    print('n components:', int(super_index.max()) + 1)
    print("Total python wrapper execution time {:.0f} s\n\n".format(exec_time))

    # Flip
    X[:, :3] = flip_coords(X[:, :3])

    # Optionally visualize after partitioning
    coords = X[:, :3]  # n x 3 float array already in the script
    labels = super_index.astype(np.int64)
    visualize_open3d(coords, first_edge, target, labels=labels)

    ###
    # Write colored PLY with x,y,z and RGB only
    base, ext = os.path.splitext(ply_file)
    out_arg = getattr(args, 'out', None)
    if out_arg:
        # if user passed a directory, place output inside it
        if os.path.isdir(out_arg):
            out_path = os.path.join(out_arg, os.path.basename(base) + '_seg.ply')
        else:
            # assume user provided a filepath; ensure parent dir exists
            parent = os.path.dirname(out_arg)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            out_path = out_arg
    else:
        out_path = base + '_seg.ply'

    write_colored_ply(X, super_index, out_path, feature_names=FEATURE_KEYS)