import cv2
import numpy as np

# Load a distorted image
image_path = "calibration_images/Photo from 2024-11-24 13-38-13.364099.jpeg"  # Replace with your image path
distorted_img = cv2.imread(image_path)

# Default values for camera matrix and distortion coefficients
# Replace these with your calibration data if available
camera_matrix = np.array([[640, 0, 320],
                          [0, 480, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)  # Default distortion coefficients

# Function to update the undistorted image based on sliders
def update_image(*args):
    fx = cv2.getTrackbarPos('Focal Length X', 'Adjust Parameters') / 10.0
    fy = cv2.getTrackbarPos('Focal Length Y', 'Adjust Parameters') / 10.0
    cx = cv2.getTrackbarPos('Center X', 'Adjust Parameters')
    cy = cv2.getTrackbarPos('Center Y', 'Adjust Parameters')
    k1 = cv2.getTrackbarPos('k1', 'Adjust Parameters') / 100.0 - 1.0
    k2 = cv2.getTrackbarPos('k2', 'Adjust Parameters') / 100.0 - 1.0
    k3 = cv2.getTrackbarPos('k3', 'Adjust Parameters') / 100.0 - 1.0
    k4 = cv2.getTrackbarPos('k4', 'Adjust Parameters') / 100.0 - 1.0

    # Update camera matrix and distortion coefficients
    updated_camera_matrix = np.array([[fx, 0, cx],
                                       [0, fy, cy],
                                       [0, 0, 1]], dtype=np.float32)
    updated_dist_coeffs = np.array([k1, k2, k3, k4], dtype=np.float32)

    # Undistort the image
    h, w = distorted_img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(updated_camera_matrix, updated_dist_coeffs, 
                                                     np.eye(3), updated_camera_matrix, (w, h), cv2.CV_16SC2)
    undistorted_img = cv2.remap(distorted_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Combine the distorted and undistorted images for comparison
    combined = np.hstack((distorted_img, undistorted_img))
    cv2.imshow('Comparison', combined)

# Create a window for sliders
cv2.namedWindow('Adjust Parameters')
h, w = distorted_img.shape[:2]

# Add sliders for the camera matrix and distortion coefficients
cv2.createTrackbar('Focal Length X', 'Adjust Parameters', 800, 2000, update_image)
cv2.createTrackbar('Focal Length Y', 'Adjust Parameters', 800, 2000, update_image)
cv2.createTrackbar('Center X', 'Adjust Parameters', w // 2, w, update_image)
cv2.createTrackbar('Center Y', 'Adjust Parameters', h // 2, h, update_image)
cv2.createTrackbar('k1', 'Adjust Parameters', 50, 200, update_image)
cv2.createTrackbar('k2', 'Adjust Parameters', 50, 200, update_image)
cv2.createTrackbar('k3', 'Adjust Parameters', 50, 200, update_image)
cv2.createTrackbar('k4', 'Adjust Parameters', 50, 200, update_image)

# Show the initial image
update_image()

# Wait until the user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
