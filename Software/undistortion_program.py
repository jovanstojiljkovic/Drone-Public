import cv2
import numpy as np

# Directly paste the camera matrix (K) and distortion coefficients (D) here
K = np.array([[1.05859732e+03, 0.00000000e+00, 9.84148261e+02],
              [0.00000000e+00, 1.05736875e+03, 5.52471341e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        
D = np.array([[-0.10407987],
              [ 0.05947105],
              [-0.10405355],
              [ 0.04506298]])

# Load the test image
img = cv2.imread("C:/Users/krake/Desktop/calibration_images/Image_1.jpg")  # Replace with your image path
if img is None:
    raise ValueError("Image not found. Please check the path!")

# Undistort the image using the hardcoded parameters
h, w = img.shape[:2]
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(img, K, D, None, new_camera_matrix)

# Display the original and undistorted images
cv2.imshow("Original Image", img)
cv2.imshow("Undistorted Image", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
