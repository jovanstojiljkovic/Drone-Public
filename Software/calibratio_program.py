import cv2
import numpy as np
import glob
import  os

print("calibration_images", os.getcwd())

CHECKERBOARD = (7, 5)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                    cv2.fisheye.CALIB_CHECK_COND + \
                    cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane
images = glob.glob('C:/Users/krake/Desktop/calibration_images/*.jpg')
print("Found images:", images)


gray = None  # Initialize gray outside the loop

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load {fname}. Skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
        
        # Add this block for debugging
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)  # Wait for 500 ms to view the image with corners drawn
        cv2.destroyAllWindows()
        
    else:
        print(f"Checkerboard not found in {fname}. Skipping.")

# Ensure at least one valid set of points is collected
if len(objpoints) > 0 and gray is not None:
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs, tvecs = [], []

    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print(f"Found {len(objpoints)} valid images for calibration")
    print(f"K =\n{K}")
    print(f"D =\n{D}")
else:
    print("No valid images for calibration. Ensure checkerboard is visible in photos.")
    
