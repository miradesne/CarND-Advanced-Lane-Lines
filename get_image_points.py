import numpy as np
import cv2
import glob
import pickle

image_dir = "camera_cal"

number_of_h_corners = 9
number_of_v_corners = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((number_of_v_corners*number_of_h_corners,3), np.float32)
objp[:,:2] = np.mgrid[0:number_of_h_corners, 0:number_of_v_corners].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob(image_dir + '/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (number_of_h_corners,number_of_v_corners), None)

	# If found, add object points, image points
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)

		# Draw and display the corners
		cv2.drawChessboardCorners(img, (number_of_h_corners,number_of_v_corners), corners, ret)
		write_name = image_dir+'/corners_found'+str(idx)+'.jpg'
		cv2.imwrite(write_name, img)
		# cv2.imshow('img', img)
		# cv2.waitKey(10)

cal_dict = {}
cal_dict["objpoints"] = objpoints
cal_dict["imgpoints"] = imgpoints

with open('calibration_points.p','wb') as f:
	pickle.dump(cal_dict,f)

# cv2.destroyAllWindows()