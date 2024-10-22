import rclpy
from rclpy.node import Node
import cv2
import torch
import numpy as np
import time
# import matplotlib  # Uncomment if needed for colormap

from depth_anything_v2.dpt import DepthAnythingV2


class DepthCameraNode(Node):
    def __init__(self):
        super().__init__('depth_camera_node')

        # Set the device (CUDA, MPS, or CPU)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Set up model configuration
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vits'  # You can change this to 'vitb', 'vitl', or 'vitg' if required
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location=self.DEVICE))
        self.model = self.model.to(self.DEVICE).eval()

        # Open webcam for live video capture
        self.cap = cv2.VideoCapture(0)  # 0 is usually the default camera index
        if not self.cap.isOpened():
            self.get_logger().error("Error: Could not open webcam.")
            self.destroy_node()
            return

        # Colormap for depth visualization (if you want to add it later)
        # self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')

        # To calculate average latency
        self.total_latency = 0
        self.frame_count = 0

        # Create a timer to call the function every frame
        self.timer = self.create_timer(0.03, self.timer_callback)  # Call every 30ms (~33fps)

    def preprocess_frame(self, frame):
        # Resize the frame to a smaller size, e.g., 256x256, for faster processing
        resized_frame = cv2.resize(frame, (256, 256))
        return resized_frame

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Error: Failed to capture frame.")
            return

        # Measure start time
        start_time = time.time()

        # Preprocess the frame
        frame = self.preprocess_frame(frame)

        # Infer depth map from the current frame
        depth = self.model.infer_image(frame)  # HxW raw depth map in numpy

        # Normalize depth to range [0, 255]
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)

        # Measure end time and calculate latency
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        # Display the latency on the OpenCV window
        latency_text = f"Latency: {latency:.4f} ms"
        cv2.putText(depth_normalized, latency_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Live Video and Depth Map', depth_normalized)

        # Break the loop if 'q' is pressed (works within ROS 2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Shutting down...")
            self.cap.release()
            cv2.destroyAllWindows()
            self.destroy_node()

    def destroy_node(self):
        self.get_logger().info("Closing the camera and stopping the node.")
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
