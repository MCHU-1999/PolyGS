# python
import numpy as np
from scipy.spatial import cKDTree
from pycut_pursuit import cp_d0_dist
import matplotlib.pyplot as plt

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

# Example: random 2D points
rng = np.random.RandomState(0)
n = 500
points = rng.randn(n, 2) * [1.0, 0.4] + [0.5, -0.5]

# Build k-NN graph
k = 8
# k = 50 
tree = cKDTree(points)
dists, inds = tree.query(points, k=k + 1)  # first is self
dists = dists[:, 1:]
inds = inds[:, 1:]

# Make directed neighbor lists
neigh = [list(map(int, inds[i])) for i in range(n)]
first_edge, target = build_forward_star(n, neigh)

# Edge weights: use distance-based weight (positive)
edge_weights = np.concatenate([dists[i] for i in range(n)]).astype(np.float32)

# Node weights and coord weights
vert_weights = np.ones(n, dtype=np.float32)
D = 2
x = np.asfortranarray(points.T.astype(np.float32))  # shape (D, n), Fortran order
coor_weights = np.ones(D, dtype=np.float32)

# Call cp_d0_dist
super_index, x_c, cluster, edges, times = cp_d0_dist(
    D,
    x,
    first_edge.astype(np.uint32),
    target.astype(np.uint32),
    edge_weights=edge_weights,
    vert_weights=vert_weights,
    coor_weights=coor_weights,
    min_comp_weight=10,      # minimum superpoint size
    cp_dif_tol=1e-2,
    cp_it_max=20,
    split_damp_ratio=0.7,
    verbose=False,
    max_num_threads=0,       # auto
    balance_parallel_split=True,
    compute_Time=True,
    compute_List=True,
    compute_Graph=True
)

print("n nodes:", n)
print("n components:", int(super_index.max()) + 1)

# Visualize (2D)
plt.scatter(points[:, 0], points[:, 1], c=super_index, s=8, cmap='tab20')
plt.title(f'Partition: {int(super_index.max())+1} components')
plt.show()