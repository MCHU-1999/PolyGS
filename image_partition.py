import numpy as np
import cv2
from colmap_io import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text, qvec2rotmat
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.interpolate import NearestNDInterpolator

from pycut_pursuit import cp_d0_dist


def build_forward_star(n, neigh_lists):
    """Build forward star representation from neighbor lists."""
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


def create_pixel_connectivity(h, w, connectivity=4):
    """Create neighbor lists for image pixels using 4 or 8-connectivity.
    
    Args:
        h, w: Image height and width
        connectivity: 4 or 8 for neighborhood type
        
    Returns:
        neigh_lists: List of neighbor lists for each pixel
        valid_mask: Boolean mask indicating valid pixels
    """
    neigh_lists = []
    valid_mask = np.ones((h, w), dtype=bool)
    
    # Define neighbor offsets
    if connectivity == 4:
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
    elif connectivity == 8:
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0),   # 4-connectivity
                   (1, 1), (1, -1), (-1, 1), (-1, -1)]  # diagonals
    else:
        raise ValueError("connectivity must be 4 or 8")
    
    for y in range(h):
        for x in range(w):
            neighbors = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                # Check bounds
                if 0 <= ny < h and 0 <= nx < w:
                    neighbor_idx = ny * w + nx
                    neighbors.append(neighbor_idx)
            neigh_lists.append(neighbors)
    
    return neigh_lists, valid_mask


def load_depth(depth_path: str):
    """Loads depth map from .npy files."""
    depth = np.load(depth_path)
    return depth

def load_normal(normal_path: str):
    """Loads normal map from .npy files."""
    normals = np.load(normal_path)
    corrected_normal = np.transpose(normals, (1, 2, 0))

    image_path = normal_path.split('.npy')[0] + '.jpg'
    rendered_normal = Image.open(image_path)

    return corrected_normal, rendered_normal

def create_features(depth, normals, K):
    """
    Identifies planes by clustering (nx, ny, nz, d) vectors.
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 1. Back-project pixels to 3D Camera Coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    points_3d = np.stack((x, y, z), axis=-1)

    # 2. Calculate plane constant 'd' where n·P + d = 0
    # d = -(nx*x + ny*y + nz*z)
    d_values = -np.sum(normals * points_3d, axis=-1)

    # 3. Form 4D feature vectors: [nx, ny, nz, d]
    features = np.concatenate([normals, d_values[..., None]], axis=-1)
    features_flat = features.reshape(-1, 4)

    return features_flat


def partition_image_features(features, h, w, min_comp_weight=50, connectivity=4, 
                           feature_weights=None, verbose=False):
    """
    Partition image pixels based on 4D features using cp_d0_dist clustering.
    
    Args:
        features: (h*w, 4) array of features [nx, ny, nz, d]
        h, w: Image dimensions
        min_comp_weight: Minimum component size (pixels)
        connectivity: 4 or 8 for pixel connectivity
        feature_weights: Weights for features, defaults to [1.0, 1.0, 1.0, 0.1]
        verbose: Print debug information
        
    Returns:
        mask: (h, w) array with cluster labels (-1 for invalid pixels)
    """
    if feature_weights is None:
        # [nx, ny, nz, d] - normals are more important than plane constant
        feature_weights = np.array([1.0, 1.0, 1.0, 0.1], dtype=np.float32)
    
    # Filter out invalid features (NaN or infinite values)
    valid_mask = np.isfinite(features).all(axis=1)
    valid_mask = valid_mask.reshape(h, w)
    
    # Create pixel connectivity graph
    print("Building pixel connectivity graph...")
    neigh_lists, _ = create_pixel_connectivity(h, w, connectivity=connectivity)
    
    # Filter features and create mapping for valid pixels only
    features_flat = features.reshape(h * w, -1)
    valid_indices = np.where(valid_mask.flatten())[0]
    valid_features = features_flat[valid_indices]
    
    print(f"Valid pixels: {len(valid_indices)} / {h * w}")
    
    # Create mapping from original pixel index to valid pixel index
    old_to_new = {}
    for new_idx, old_idx in enumerate(valid_indices):
        old_to_new[old_idx] = new_idx
    
    # Filter neighbor lists to only include valid pixels
    valid_neigh_lists = []
    for i, old_idx in enumerate(valid_indices):
        neighbors = []
        for neighbor_idx in neigh_lists[old_idx]:
            if neighbor_idx in old_to_new:
                neighbors.append(old_to_new[neighbor_idx])
        valid_neigh_lists.append(neighbors)
    
    # Build forward star representation
    n_valid = len(valid_indices)
    first_edge, target = build_forward_star(n_valid, valid_neigh_lists)
    print(f"Graph built: {n_valid} vertices, {target.shape[0]} edges")
    
    # Prepare features for cp_d0_dist (transpose to Fortran order)
    x = np.asfortranarray(valid_features.T.astype(np.float32))
    
    # Create edge weights (uniform for now, could be based on feature similarity)
    edge_weights = np.ones(target.shape[0], dtype=np.float32)
    
    # Run cp_d0_dist clustering
    print("Running cp_d0_dist clustering...")
    super_index, x_c, cluster, edges, times = cp_d0_dist(
        valid_features.shape[1],  # D = 4 (nx, ny, nz, d)
        x,
        first_edge.astype(np.uint32),
        target.astype(np.uint32),
        edge_weights=edge_weights,
        vert_weights=None,
        coor_weights=feature_weights,
        min_comp_weight=min_comp_weight,
        cp_dif_tol=1e-2,
        cp_it_max=20,
        split_damp_ratio=0.7,
        verbose=verbose,
        max_num_threads=0,
        balance_parallel_split=True,
        compute_Time=True,
        compute_List=True,
        compute_Graph=True
    )
    
    # Create full mask with background
    mask = np.full(h * w, -1, dtype=int)  # -1 for invalid/background
    mask[valid_indices] = super_index
    mask = mask.reshape(h, w)
    
    print(f"Found {int(super_index.max()) + 1} clusters")
    
    return mask


# --- Execution ---
if __name__ == "__main__":
    BASE_DIR = "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn"
    DEPTH_DIR = f"{BASE_DIR}/scaled_depth"
    NORMAL_DIR = f"{BASE_DIR}/mono_normal"
    WEIGHTS = np.array([
        20, 20, 20, 1
    ], dtype=np.float32)

    # 1. Load COLMAP Camera Data
    # Adjust these paths to your sparse folder
    cameras = read_intrinsics_binary(f"{BASE_DIR}/sparse/0/cameras.bin")
    images = read_extrinsics_binary(f"{BASE_DIR}/sparse/0/images.bin")

    # 2. Extract K matrix for the first camera
    cam_id = list(cameras.keys())[0]
    cam = cameras[cam_id]

    assert cam.model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
    fx, fy, cx, cy = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 3. Load your .npy Maps
    depth_map = load_depth(f"{DEPTH_DIR}/000001.npy")
    normal_map, rendered_normal = load_normal(f"{NORMAL_DIR}/000001.npy")

    # 4. Generate Planar Masks
    features = create_features(depth_map, normal_map, K)
    h, w = depth_map.shape
    mask = partition_image_features(features, h, w, feature_weights=WEIGHTS, min_comp_weight=500, verbose=False)

    # 5. Visualize Results
    plt.figure(figsize=(18, 6))  # Make wider to accommodate 3 subplots

    plt.subplot(1, 3, 1)
    plt.title("Input Depth")
    plt.imshow(depth_map, cmap='jet')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Rendered Normal")
    plt.imshow(rendered_normal)  # PIL Image displays directly
    plt.axis('off')  # Remove axes for cleaner normal display

    plt.subplot(1, 3, 3)
    plt.title("Detected Planes (Masks)")
    
    # Check what labels we have
    unique_labels = np.unique(mask)
    print(f"Unique labels: {unique_labels}")
    print(f"Label counts: {[(label, np.sum(mask == label)) for label in unique_labels]}")

    # Set background/noise (-1) to 0 for black color
    mask_vis = mask.copy().astype(float)
    mask_vis[mask == -1] = 0

    # Get valid cluster labels (excluding noise)
    valid_labels = unique_labels[unique_labels >= 0]
    n_clusters = len(valid_labels)

    if n_clusters > 0:
        # For many clusters, generate random colors
        np.random.seed(42)  # For reproducible colors
        colors_list = ['black']  # Background
        # Generate random bright colors for each cluster
        for i in range(n_clusters):
            color = np.random.rand(3,)
            # Make colors more vibrant by ensuring at least one channel is > 0.7
            max_idx = np.argmax(color)
            color[max_idx] = max(color[max_idx], 0.7)
            colors_list.append(tuple(color))
            
        random_cmap = ListedColormap(colors_list)
        
        # Map cluster labels to sequential indices for visualization
        for i, label in enumerate(valid_labels):
            mask_vis[mask == label] = i + 1
        
        plt.imshow(mask_vis, cmap=random_cmap, vmin=0, vmax=n_clusters)
        plt.colorbar(label=f'{n_clusters} planes detected')
    else:
        # No clusters found, just show black background
        plt.imshow(mask_vis, cmap='gray')
        plt.colorbar()

    plt.tight_layout()  # Improve spacing
    plt.show()