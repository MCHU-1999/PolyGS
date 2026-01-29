# python partition.py --ply example_pointcloud/dtu_scan24.ply
import argparse
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from pycut_pursuit import cp_d0_dist
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from plyfile import PlyData
except Exception:
    raise RuntimeError("Please install 'plyfile' (pip install plyfile) to read PLY files")

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


# Parse arguments
parser = argparse.ArgumentParser(description='Partition a PLY pointcloud with cp_d0_dist')
parser.add_argument('--ply', required=True, help='Path to input PLY file')
parser.add_argument('--out', help='Path to output PLY file')
parser.add_argument('--k', type=int, default=8, help='k for k-NN graph')
parser.add_argument('--min-comp', type=float, default=10.0, help='min component weight (points)')
parser.add_argument('--spatial-weight', type=float, default=1.0, help='weight for coordinate dimensions')
parser.add_argument('--max-it', type=int, default=20, help='cp_d0_dist max iterations')
parser.add_argument('--plot', action='store_true', help='Show 3D plot of partition')
args = parser.parse_args()


ply_file = args.ply
pc = PlyData.read(ply_file)
# pd.elements is a list of PlyElement objects; check names properly
elements = [e.name for e in pc.elements]
if 'vertex' not in elements:
    raise RuntimeError(f"PLY has no 'vertex' element. Found elements: {elements}")

data = pc['vertex'].data
names = data.dtype.names

# Coordinates
coords = np.vstack([data[name] for name in ('x', 'y', 'z')]).T.astype(np.float32)

# Build features list excluding f_rest_*
known = ['nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
feat_keys = [k for k in known if k in names]
# include other f_* except f_rest_*
for k in names:
    if k.startswith('f_rest_'):
        continue
    if k in ('x','y','z'):
        continue
    if k in feat_keys:
        continue
    if k.startswith('f_'):
        feat_keys.append(k)

if feat_keys:
    feats = np.vstack([data[k] for k in feat_keys]).T.astype(np.float32)
else:
    feats = np.empty((coords.shape[0], 0), dtype=np.float32)

n = coords.shape[0]
print(f'Loaded {n} points, using features: {feat_keys}')

# Prepare x matrix: coords centered + features
pos_offset = coords.mean(axis=0)
coords_centered = (coords - pos_offset).astype(np.float32)
if feats.shape[1] > 0:
    X = np.hstack((coords_centered, feats))
else:
    X = coords_centered
D = X.shape[1]
x = np.asfortranarray(X.T.astype(np.float32))

# k-NN graph on coordinates
k = args.k
tree = cKDTree(coords)
dists, inds = tree.query(coords, k=k + 1)
dists = dists[:, 1:]
inds = inds[:, 1:]
neigh = [list(map(int, inds[i])) for i in range(n)]
first_edge, target = build_forward_star(n, neigh)

# Edge weights: exponential kernel
flat_dists = np.concatenate([dists[i] for i in range(n)]).astype(np.float32)
mean_d = flat_dists.mean() if flat_dists.size else 1.0
edge_weights = np.exp(-flat_dists / (mean_d + 1e-12)).astype(np.float32)

# Vertex weights: atomic points -> 1
vert_weights = np.ones(n, dtype=np.float32)

# Coordinate weights
coor_weights = np.ones(D, dtype=np.float32)
coor_weights[:3] *= args.spatial_weight

# Run partition
start_time = time.time()
super_index, x_c, cluster, edges, times = cp_d0_dist(
    D,
    x,
    first_edge.astype(np.uint32),
    target.astype(np.uint32),
    edge_weights=edge_weights,
    vert_weights=vert_weights,
    coor_weights=coor_weights,
    min_comp_weight=args.min_comp,
    cp_dif_tol=1e-2,
    cp_it_max=args.max_it,
    split_damp_ratio=0.7,
    verbose=False,
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


# Map labels to colors using a colormap
cmap = cm.get_cmap('tab20')
labels = super_index.astype(np.int64)
num_labels = int(labels.max()) + 1
# normalize labels into [0,1]
norm = labels / max(1, num_labels - 1)
colors = (cmap(norm)[:, :3] * 255).astype(np.uint8)

# Build structured array for Ply
vertex = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
vertex['x'] = coords[:, 0]
vertex['y'] = coords[:, 1]
vertex['z'] = coords[:, 2]
vertex['red'] = colors[:, 0]
vertex['green'] = colors[:, 1]
vertex['blue'] = colors[:, 2]

# Write binary little endian ply
try:
    from plyfile import PlyData, PlyElement
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(out_path)
    print(f'Wrote colored PLY to: {out_path}')
except Exception as e:
    print('Failed to write PLY:', e)

# Plot (3D or 2D fallback)
if args.plot:
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=super_index, s=1, cmap='tab20')
        plt.title(f'Partition: {int(super_index.max())+1} components')
        plt.show()
    except Exception:
        plt.scatter(coords[:, 0], coords[:, 1], c=super_index, s=1, cmap='tab20')
        plt.title(f'Partition: {int(super_index.max())+1} components (2D projection)')
        plt.show()