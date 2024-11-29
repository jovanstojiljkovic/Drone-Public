import cv2
import torch
import numpy as np
import matplotlib
import time
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits'  # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location=DEVICE))
model = model.to(DEVICE).eval()

# Colormap for depth visualization
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

def preprocess_image(image):
    # Resize the image for consistent processing
    resized_image = cv2.resize(image, (640, 480))
    return resized_image

def process_single_image(image_path):
    print(f"Processing image: {image_path}")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}.")
        return

    # Preprocess the image
    image = preprocess_image(image)

    # Measure start time
    start_time = time.time()

    # Infer depth map
    depth = model.infer_image(image)  # HxW raw depth map in numpy

    # Normalize depth to range [0, 255]
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

    # Create a binary mask for the foreground
    threshold = np.percentile(depth, 76)  # Set threshold for background removal
    foreground_mask = depth >= threshold

    # Apply the mask to the original image
    masked_image = image.copy()
    masked_image[~foreground_mask] = 0  # Set background pixels to black

    # Detect edges in the masked image
    gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_masked, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    # Create a copy of the masked image to draw the outline
    outlined_image = masked_image.copy()

    if largest_contour is not None:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Draw the largest face outline
        cv2.drawContours(outlined_image, [approx_polygon], -1, (0, 255, 0), 3)  # Green outline

    # Measure end time
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Processing completed in {latency:.4f} ms.")

    # Combine original, depth map, masked foreground, and outlined image for display
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
    combined_result = cv2.hconcat([
        image, split_region, depth_colored[:, :, ::-1], split_region, outlined_image
    ])  # Convert depth map to BGR for OpenCV

    # Display the combined result
    cv2.imshow('Original, Depth Map, Masked, and Outlined Cuboid Face', combined_result)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

# Path to the specific image
image_path = "test_images/Direct1.jpeg"  # Replace with the path to your specific image

process_single_image(image_path)
