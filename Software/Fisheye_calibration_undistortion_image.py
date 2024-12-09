import yaml
import cv2
import numpy as np
import glob

DIM=(1920, 1080)
balance=0
dim2=None
dim3=None
K = np.array([[1.04137480e+03, 0.00000000e+00, 9.89367620e+02],
              [0.00000000e+00, 1.04230431e+03, 5.47297766e+02],
              [0.00000000e+00, 0.00000000e+00 ,1.00000000e+00]])
D = np.array([[-0.09235215],
              [-0.01144683],
              [-0.00626767], 
              [ 0.00810741]])

img = cv2.imread("C:/Users/krake/Desktop/calibration_images/Image_14.jpg")
dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
if not dim2:
    dim2 = dim1
if not dim3:
    dim3 = dim1
scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. 
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


cv2.imshow("img1", img)
cv2.imshow("undistorted1", undistorted_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
