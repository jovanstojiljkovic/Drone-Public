
import cv2
import os
import torch
import numpy as np
import matplotlib
import time
# import torch_tensorrt

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
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cuda'))
model = model.to(DEVICE).eval()

# Convert model to TensorRT for better performance
#model_trt = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((1, 3, 256, 256))], enabled_precisions={torch.float32})

# Open webcam for live video capture
cap = cv2.VideoCapture(0)  # 0 is usually the default camera index

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Colormap for depth visualization
# cmap = matplotlib.colormaps.get_cmap('Spectral_r')

def preprocess_frame(frame):
    # Resize the frame to a smaller size, e.g., 256x256, for faster processing
    resized_frame = cv2.resize(frame, (256, 256))
    return resized_frame

# To calculate average latency
total_latency = 0
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Measure start time
    start_time = time.time()

    frame = preprocess_frame(frame)
    # Infer depth map from the current frame
    depth = model.infer_image(frame)  # HxW raw depth map in numpy

    # Normalize depth to range [0, 255]
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

    # Apply colormap to the depth image
    # depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)

    # Add a split region between images (white space)
    #split_region = np.ones((frame.shape[0], 50, 3), dtype=np.uint8) * 255

    # Combine the original frame and the depth-colored frame side by side
    #combined_result = cv2.hconcat([frame, split_region, depth_normalized[:, :, ::-1]])  # Convert to BGR for OpenCV

    # Measure end time and calculate latency
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
   
    # Display the latency on the OpenCV window
    latency_text = f"Latency: {latency:.4f} ms"
    cv2.putText(depth_normalized, latency_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Live Video and Depth Map', depth_normalized)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
