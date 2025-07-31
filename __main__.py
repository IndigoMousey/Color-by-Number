from ColorQuantization import *
from edges import *

def rgb_to_label_map(img, palette):
    """Convert RGB image to label map using palette index."""
    H, W, _ = img.shape
    label_map = np.zeros((H, W), dtype=np.uint8)
    for i, color in enumerate(palette):
        mask = np.all(img == color, axis=2)
        label_map[mask] = i
    return label_map

def apply_edges_to_image(img, edge_mask):
    """Draw black edges over the image."""
    img_with_edges = img.copy()
    img_with_edges[edge_mask == 255] = [0, 0, 0]
    return img_with_edges

images = ["oriole.jpg", "bird.jpg"]
for img_path in images:
    print(f"Processing: {img_path}")
    original, quantized = kmeansClustering(img_path, nClusters=10)

    # Re-extract palette and label map
    rows, cols, _ = quantized.shape
    flat_img = quantized.reshape((-1, 3))
    unique_colors, inverse = np.unique(flat_img, axis=0, return_inverse=True)
    label_map = inverse.reshape((rows, cols))

    # Clean up noisy regions
    clean_labels = remove_small_regions(label_map, min_size=30)

    # Generate edge mask from cleaned labels
    edge_mask = get_quantized_edges(clean_labels)

    # Replace label map with cleaned palette
    clean_rgb = unique_colors[clean_labels]

    # Reshape back to image
    clean_rgb_img = clean_rgb.reshape((rows, cols, 3))

    # Apply black edges
    outlined_img = apply_edges_to_image(clean_rgb_img, edge_mask)

    # Display side-by-side
    side_by_side(original, outlined_img)

    # Optional: save output
    # out_path = os.path.splitext(img_path)[0] + "_color_by_number.png"
    # cv2.imwrite(out_path, outlined_img[:, :, ::-1])  # Convert RGB to BGR for OpenCV
    # print(f"Saved: {out_path}")
