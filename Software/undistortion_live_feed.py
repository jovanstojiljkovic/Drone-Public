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

# Open the camera feed
cap = cv2.VideoCapture(0)  # 0 is usually the default camera, so your main camera on you laptop

if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Get the frame dimensions
    h, w = frame.shape[:2]
    
    # Undistort the frame using the calibration parameters
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, K, D, None, new_camera_matrix)
    
    # Display the original and undistorted frames
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Undistorted Frame", undistorted_frame)

    # If you press q the camera feed will close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
