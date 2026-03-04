import numpy as np
import cv2
from colmap_io import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text, qvec2rotmat
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.interpolate import NearestNDInterpolator


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

def get_planar_masks(depth, normals, K, eps=0.05, min_samples=1000):
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

    # 4. Filter out invalid depth (0 or inf)
    valid_mask = (depth.flatten() > 0) & (np.isfinite(depth.flatten()))
    valid_features = features_flat[valid_mask]

    # 5. Spatial Downsampling for DBSCAN speed
    # We cluster every 10th pixel to find the plane parameters
    step = 10
    train_features = valid_features[::step]

    print(f"Clustering {len(train_features)} points...")
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(train_features)
    
    # 6. Map labels back to full resolution image
    full_labels = np.full(h * w, -1, dtype=int)
    # For a quick visualization, we only show the trained points
    valid_indices = np.where(valid_mask)[0]
    train_indices = valid_indices[::step]
    full_labels[train_indices] = db.labels_
    
    return full_labels.reshape(h, w)

def broadcast_labels_dilation(mask, iterations=5):
    """
    Uses iterative dilation to fill gaps between downsampled points.
    """
    # mask is (H, W) with -1 for empty space
    # We shift labels to be positive for CV2 compatibility
    offset_mask = (mask + 1).astype(np.uint16)
    kernel = np.ones((3,3), np.uint8)
    
    # Dilation will expand the colored pixels into the 0 (previously -1) pixels
    # We use a custom loop to ensure we don't overwrite existing labels
    for _ in range(iterations):
        dilated = cv2.dilate(offset_mask, kernel)
        # Only update pixels that are currently 'background' (0)
        mask_zeros = (offset_mask == 0)
        offset_mask[mask_zeros] = dilated[mask_zeros]
        
    return offset_mask.astype(int) - 1


# --- Execution ---
if __name__ == "__main__":
    BASE_DIR = "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn"
    DEPTH_DIR = f"{BASE_DIR}/scaled_depth"
    NORMAL_DIR = f"{BASE_DIR}/mono_normal"

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
    depth_map = load_depth(f"{DEPTH_DIR}/000051.npy")
    normal_map, rendered_normal = load_normal(f"{NORMAL_DIR}/000051.npy")

    # 4. Generate Planar Masks
    mask = get_planar_masks(depth_map, normal_map, K, 0.05, 1000)
    mask = broadcast_labels_dilation(mask, 2)

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
    plt.imshow(mask, cmap='Set1_r')
    plt.colorbar()


    plt.tight_layout()  # Improve spacing
    plt.show()