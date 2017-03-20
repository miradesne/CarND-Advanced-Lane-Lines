import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read in the saved objpoints and imgpoints
# dist_pickle = pickle.load( open( "calibration_points.p", "rb" ) )
# objpoints = dist_pickle["objpoints"]
# imgpoints = dist_pickle["imgpoints"]

# # Read in an image
# img = cv2.imread('camera_cal/calibration3.jpg')

# a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# img_size = (img.shape[1], img.shape[0])
	# Use cv2.calibrateCamera() and cv2.undistort()
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	return undist

def warp(img, src, dst, img_size):
	trans_m = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(img, trans_m, img_size, flags=cv2.INTER_LINEAR)
	return warped


# undistorted = cal_undistort(img, objpoints, imgpoints)

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(undistorted)
# ax2.set_title('Undistorted Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()

# input('press any key to the next step')