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
import networkx as nx
from sklearn.decomposition import PCA

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
    20, 20, 20,
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
    """

    coords = X[:, :3]
    finite_mask = np.isfinite(coords).all(axis=1)
    if finite_mask.all():
        return X

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

def compute_superpoint_properties(X, super_index):
    """Compute centroid and normal for each superpoint using PCA.
    
    Returns:
    - centroids: (n_superpoints, 3) array of superpoint centroids
    - normals: (n_superpoints, 3) array of superpoint normal vectors
    - sizes: (n_superpoints,) array of point counts per superpoint
    """
    n_superpoints = int(super_index.max()) + 1
    coords = X[:, :3]
    
    centroids = np.zeros((n_superpoints, 3))
    normals = np.zeros((n_superpoints, 3))
    sizes = np.zeros(n_superpoints, dtype=int)
    for sp_id in range(n_superpoints):
        mask = super_index == sp_id
        if not mask.any():
            continue
            
        sp_coords = coords[mask]
        sizes[sp_id] = sp_coords.shape[0]
        
        # Compute centroid
        centroids[sp_id] = sp_coords.mean(axis=0)
        
        # Compute normal
        if X.shape[1] >= 6:  # has nx, ny, nz
            normal = X[mask, 3:6].mean(axis=0)
            # Orient normal consistently (e.g., prefer positive Z when possible)
            if normal[2] < 0:  # if pointing down, flip to point up
                normal = -normal
            elif normal[2] == 0:  # if horizontal, prefer positive Y
                if normal[1] < 0:
                    normal = -normal
            elif normal[1] == 0 and normal[0] < 0:  # if along X, prefer positive
                normal = -normal              
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            normals[sp_id] = normal
        else:
            raise Exception("No nx, ny, or nz in this array.")

    return centroids, normals, sizes


def build_superpoint_adjacency_graph(X, super_index, first_edge, target, distance_threshold=None):
    """Build NetworkX graph of superpoint adjacencies.
    
    Returns:
    - G: NetworkX graph where nodes are superpoint IDs
    """
    n_superpoints = int(super_index.max()) + 1
    n_points = X.shape[0]
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n_superpoints))
    
    # Find adjacent superpoints through point-level edges
    adjacencies = set()
    
    for i in range(n_points):
        sp_i = super_index[i]
        # Check all neighbors of point i
        for e in range(first_edge[i], first_edge[i + 1]):
            j = target[e]
            if j < n_points:  # valid neighbor
                sp_j = super_index[j]
                if sp_i != sp_j:  # different superpoints
                    adjacencies.add((min(sp_i, sp_j), max(sp_i, sp_j)))
    
    # Add edges to graph
    G.add_edges_from(adjacencies)
    
    # Optionally filter by distance threshold (using minimum inter-cluster distance)
    if distance_threshold is not None:
        coords = X[:, :3]
        edges_to_remove = []
        
        # For each superpoint pair, find minimum distance between their points
        # Use existing point-level adjacency for efficiency
        sp_min_distances = {}
        
        for i in range(n_points):
            sp_i = super_index[i]
            # Check distances to neighbors in different superpoints
            for e in range(first_edge[i], first_edge[i + 1]):
                j = target[e]
                if j < n_points:
                    sp_j = super_index[j]
                    if sp_i != sp_j:
                        # Distance between these two connected points
                        dist = np.linalg.norm(coords[i] - coords[j])
                        pair_key = (min(sp_i, sp_j), max(sp_i, sp_j))
                        if pair_key not in sp_min_distances:
                            sp_min_distances[pair_key] = dist
                        else:
                            sp_min_distances[pair_key] = min(sp_min_distances[pair_key], dist)
        
        # Remove edges where minimum distance exceeds threshold
        for sp_i, sp_j in G.edges():
            pair_key = (min(sp_i, sp_j), max(sp_i, sp_j))
            if pair_key in sp_min_distances:
                min_dist = sp_min_distances[pair_key]
                if min_dist > distance_threshold:
                    edges_to_remove.append((sp_i, sp_j))
            # If no direct point connection found, remove edge (shouldn't happen normally)
            else:
                edges_to_remove.append((sp_i, sp_j))
        
        G.remove_edges_from(edges_to_remove)
        if edges_to_remove:
            print(f"Removed {len(edges_to_remove)} superpoint connections exceeding distance threshold {distance_threshold}")
    
    return G
    
def merge_coplanar_superpoints(G, centroids, normals, sizes, angle_threshold_deg=20.0, min_size=5):
    """Merge connected co-planar superpoints into larger regions.
    
    Returns:
    - merge_tree: NetworkX DiGraph representing the hierarchical merging
    - final_labels: mapping from original superpoint ID to final merged region ID
    """
    angle_threshold_rad = np.deg2rad(angle_threshold_deg)
    
    # Create merge tree (directed graph: parent -> children)
    merge_tree = nx.DiGraph()
    
    # Start with all original superpoints as leaf nodes
    active_nodes = set(G.nodes())
    next_node_id = len(active_nodes)
    
    # Add original nodes to merge tree
    for node_id in active_nodes:
        merge_tree.add_node(node_id, 
                           centroid=centroids[node_id].copy(),
                           normal=normals[node_id].copy(),
                           size=sizes[node_id],
                           is_leaf=True)
    
    merged_something = True
    iteration = 0
    
    while merged_something and len(active_nodes) > 1:
        merged_something = False
        iteration += 1
        print(f"Merge iteration {iteration}: {len(active_nodes)} active regions")
        
        # Find best merge candidate
        best_pair = None
        best_score = float('inf')
        
        # Check all adjacent pairs
        for node_i in active_nodes:
            for node_j in G.neighbors(node_i):
                if node_j not in active_nodes or node_j <= node_i:
                    continue
                    
                # Get properties from merge tree
                props_i = merge_tree.nodes[node_i]
                props_j = merge_tree.nodes[node_j]
                
                normal_i = props_i['normal']
                normal_j = props_j['normal']
                size_i = props_i['size']
                size_j = props_j['size']
                
                # Skip if either region is too small
                if size_i < min_size or size_j < min_size:
                    continue
                
                # Compute angle between normals
                cos_angle = np.clip(np.abs(np.dot(normal_i, normal_j)), 0, 1)
                angle = np.arccos(cos_angle)
                
                # Check if co-planar
                if angle <= angle_threshold_rad:
                    # Score based on angle (lower is better)
                    score = angle
                    if score < best_score:
                        best_score = score
                        best_pair = (node_i, node_j)
        
        # Perform merge if found good candidate
        if best_pair is not None:
            node_i, node_j = best_pair
            merged_something = True
            
            # Get properties
            props_i = merge_tree.nodes[node_i]
            props_j = merge_tree.nodes[node_j]
            
            # Create new merged node
            merged_size = props_i['size'] + props_j['size']
            
            # Weighted average of centroids
            merged_centroid = (props_i['centroid'] * props_i['size'] + 
                             props_j['centroid'] * props_j['size']) / merged_size
            
            # Average of normals (could be improved)
            merged_normal = (props_i['normal'] + props_j['normal'])
            merged_normal = merged_normal / np.linalg.norm(merged_normal)
            
            # Add merged node to tree
            merge_tree.add_node(next_node_id,
                              centroid=merged_centroid,
                              normal=merged_normal,
                              size=merged_size,
                              is_leaf=False)
            
            # Add edges showing this node contains the merged nodes
            merge_tree.add_edge(next_node_id, node_i)
            merge_tree.add_edge(next_node_id, node_j)
            
            # Update adjacency graph: remove old nodes, add new one
            # Get all neighbors of both old nodes
            neighbors_i = set(G.neighbors(node_i))
            neighbors_j = set(G.neighbors(node_j))
            all_neighbors = (neighbors_i | neighbors_j) - {node_i, node_j}
            
            G.remove_nodes_from([node_i, node_j])
            G.add_node(next_node_id)
            G.add_edges_from([(next_node_id, neighbor) for neighbor in all_neighbors])
            
            # Update active nodes
            active_nodes.remove(node_i)
            active_nodes.remove(node_j)
            active_nodes.add(next_node_id)
            
            next_node_id += 1
            
            print(f"  Merged regions {node_i} and {node_j} -> {next_node_id-1} "
                  f"(angle: {np.rad2deg(best_score):.1f}°, size: {merged_size})")
    
    # Create final labels mapping original superpoints to final regions
    final_labels = {}
    
    def assign_labels(node_id, final_region_id):
        if merge_tree.nodes[node_id]['is_leaf']:
            final_labels[node_id] = final_region_id
        else:
            for child in merge_tree.successors(node_id):
                assign_labels(child, final_region_id)
    
    # Each active node becomes a final region
    for i, region_id in enumerate(sorted(active_nodes)):
        assign_labels(region_id, i)
    
    print(f"Final result: {len(active_nodes)} merged regions from {len(centroids)} original superpoints")
    
    return merge_tree, final_labels


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
    parser.add_argument('--k', type=int, default=8, help='k for k-NN graph')
    parser.add_argument('--min-comp', type=float, default=10.0, help='min component weight (points)')
    parser.add_argument('--max-it', type=int, default=30, help='cp_d0_dist max iterations')
    parser.add_argument('--keep-largest', type=bool, default=True, help='Keep only the largest connected component of the k-NN graph')
    parser.add_argument('--verbose', action='store_true', help='Whether to print the point cloud feature values or not')
    parser.add_argument('--merge-coplanar', action='store_true', help='Enable hierarchical merging of co-planar superpoints')
    parser.add_argument('--angle-threshold', type=float, default=5.0, help='Angle threshold (degrees) for co-planarity detection')
    parser.add_argument('--min-region-size', type=int, default=10, help='Minimum size for regions to be merged')
    parser.add_argument('--distance-threshold', type=float, default=0.05, help='Max distance between superpoint centroids to be considered adjacent')
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

    # Hierarchical merging of co-planar superpoints
    final_labels = super_index.copy()  # Default: no merging
    if args.merge_coplanar:
        print("\n=== Hierarchical Merging ===")
        # Compute superpoint properties
        print("Computing superpoint properties...")
        centroids, normals, sizes = compute_superpoint_properties(X, super_index)
        
        # Build superpoint adjacency graph
        print("Building superpoint adjacency graph...")
        sp_graph = build_superpoint_adjacency_graph(X, super_index, first_edge, target, 
                                                   distance_threshold=args.distance_threshold)
        print(f"Superpoint graph: {sp_graph.number_of_nodes()} nodes, {sp_graph.number_of_edges()} edges")
        
        # Merge co-planar superpoints
        print("Merging co-planar superpoints...")
        merge_tree, sp_to_region = merge_coplanar_superpoints(
            sp_graph, centroids, normals, sizes,
            angle_threshold_deg=args.angle_threshold,
            min_size=args.min_region_size
        )
        
        # Map points to final merged regions
        final_labels = np.array([sp_to_region[sp_id] for sp_id in super_index])
        print(f"Merged {int(super_index.max()) + 1} superpoints into {int(final_labels.max()) + 1} regions")

    # Flip coordinates for visualization
    X[:, :3] = flip_coords(X[:, :3])

    # Optionally visualize after partitioning
    coords = X[:, :3]  # n x 3 float array already in the script
    labels = final_labels.astype(np.int64)
    visualize_open3d(coords, first_edge, target, labels=labels)

    ###
    # Write colored PLY with final merged regions
    base, ext = os.path.splitext(ply_file)
    out_arg = getattr(args, 'out', None)
    if out_arg:
        # if user passed a directory, place output inside it
        if os.path.isdir(out_arg):
            suffix = '_merged.ply' if args.merge_coplanar else '_seg.ply'
            out_path = os.path.join(out_arg, os.path.basename(base) + suffix)
        else:
            # assume user provided a filepath; ensure parent dir exists
            parent = os.path.dirname(out_arg)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            out_path = out_arg
    else:
        suffix = '_merged.ply' if args.merge_coplanar else '_seg.ply'
        out_path = base + suffix

    write_colored_ply(X, final_labels, out_path, feature_names=FEATURE_KEYS)