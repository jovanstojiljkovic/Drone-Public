import cv2
import numpy as np
from itertools import combinations

# Camera matrix and distortion coefficients (example values for fisheye lens)
# Replace these with the actual calibration data for your camera
camera_matrix = np.array([[1000, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-0.03, 0.04, 0, 0], dtype=np.float32)  # Replace with your distortion coefficients

def undistort_fisheye_image(img, camera_matrix, dist_coeffs):
    """
    Undistorts a fisheye image given a camera matrix and distortion coefficients.
    """
    h, w = img.shape[:2]
    # Compute undistortion and rectification transformation map
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), camera_matrix, (w, h), cv2.CV_16SC2
    )
    # Apply the map to undistort the image
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def detect_lines_and_corners(frame):
    """
    Detect lines using Hough Transform and find corners of the cuboid.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)  # Adjust threshold if needed
    if lines is None:
        return frame, []

    # Convert lines from polar to Cartesian coordinates
    cartesian_lines = []
    for rho, theta in lines[:, 0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cartesian_lines.append(((x1, y1), (x2, y2)))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw lines in green

    # Find intersections between lines
    def line_intersection(line1, line2):
        """
        Find the intersection point of two lines.
        Each line is represented as ((x1, y1), (x2, y2)).
        """
        x1, y1, x2, y2 = *line1[0], *line1[1]
        x3, y3, x4, y4 = *line2[0], *line2[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Parallel lines

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return int(px), int(py)

    intersections = []
    for line1, line2 in combinations(cartesian_lines, 2):
        intersect = line_intersection(line1, line2)
        if intersect and 0 <= intersect[0] < frame.shape[1] and 0 <= intersect[1] < frame.shape[0]:
            intersections.append(intersect)

    # Draw corners on the frame
    for corner in intersections:
        cv2.circle(frame, corner, 5, (0, 0, 255), -1)  # Draw corners in red

    return frame, intersections

# Open webcam for live video capture
cap = cv2.VideoCapture(4)  # Use the correct camera index

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Undistort the fisheye frame
    undistorted_frame = undistort_fisheye_image(frame, camera_matrix, dist_coeffs)

    # Detect lines and corners
    processed_frame, corners = detect_lines_and_corners(frame)

    # Display the processed frame
    cv2.imshow('Cuboid Detection with Corners', processed_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

