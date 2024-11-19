import cv2
import os 

import torch
import cv2
import torch
import numpy as np
import open3d as o3d
from depth_anything_v2.dpt import DepthAnythingV2

# Set up device
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model configuration
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Choose the encoder
encoder = 'vits' # or 'vits', 'vitb', 'vitg'

# Initialize and load the model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# Read the input image
image_path1 = '../Images/Wind_Turbine_2024-Oct-20_01-24-10PM-000_CustomizedView13512378128.png'
image_path2 = '../Images/Wind_Turbine_2024-Oct-20_01-23-22PM-000_CustomizedView17927482789.png'
image_path = image_path1
raw_img = cv2.imread(image_path)

# Infer depth map
depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

# Step 1: Convert the depth map to a 3D point cloud
def depth_to_point_cloud(depth_map, color_image, fx=500, fy=500, cx=None, cy=None):
    """
    Convert a depth map to a 3D point cloud.
    
    Parameters:
    - depth_map: Depth map (HxW)
    - color_image: Original image to get colors (HxWx3)
    - fx, fy: Focal lengths (in pixels)
    - cx, cy: Principal points (in pixels, defaults to image center)
    
    Returns:
    - point_cloud: Open3D point cloud object
    """
    h, w = depth_map.shape
    if cx is None:
        cx = w / 2
    if cy is None:
        cy = h / 2
    
    # Generate grid of (u, v) pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert pixel coordinates to normalized camera coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    # Stack and flatten the coordinates
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Get color values from the original image and reshape to match the point array
    colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Open3D
    colors = colors.reshape(-1, 3) / 255.0

    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

# Step 2: Generate the 3D point cloud
depth_uint16 = depth.astype(np.float32)  # Use the original float depth values
color_image = cv2.imread(image_path)  # Ensure the color image matches the depth map dimensions
point_cloud = depth_to_point_cloud(depth_uint16, color_image)

# Step 3: Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])

# Normalize depth to 0-255 for display purposes
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

# Convert to uint8 to display it as an image
depth_uint8 = depth_normalized.astype(np.uint8)

# Display the depth image
cv2.imshow('Depth Map', depth_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()
