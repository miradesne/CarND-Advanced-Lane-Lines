import pickle 
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from calibration import cal_undistort, warp
from thresh import img_from_thresh
from utilities import showImgs, region_of_interest, draw_lane
from sliding_window import find_lanes, find_next_lanes, curvature

# load the test images 
image_dir = "test_images"
img_files = glob.glob(image_dir + '/test*.jpg')
imgs = [mpimg.imread(f) for f in img_files]
print("test images loaded.")

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "calibration_points.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

class Line():
	def __init__(self):
		self.n = 5
		# was the line detected in the last iteration?
		self.detected = False  
		# x values of the last n fits of the line
		self.recent_xfitted = np.array([])
		#average x values of the fitted line over the last n iterations
		self.bestx = None     
		self.recent_fit = np.array([])
		#polynomial coefficients averaged over the last n iterations
		self.best_fit = None  
		#radius of curvature of the line in some units
		# self.radius_of_curvature = None 
		#distance in meters of vehicle center from the line
		# self.line_base_pos = None 
		#difference in fit coefficients between last and new fits
		# self.diffs = np.array([0,0,0], dtype='float') 
		#x values for detected line pixels
		# self.allx = None  
		# #y values for detected line pixels
		# self.ally = None

	def set_x_fit(self, x_fit):
		if self.detected == False:
			self.recent_xfitted = np.array([x_fit])
		else:
			self.recent_xfitted = np.append(self.recent_xfitted, [x_fit], axis=0)
		if self.recent_xfitted.shape[0] > self.n:
			self.recent_xfitted = np.delete(self.recent_xfitted, 0, 0)
		self.bestx = np.average(self.recent_xfitted, axis=0)

	def set_fit(self, fit):
		if self.detected == False:
			self.recent_fit = np.array([fit])
		else:
			self.recent_fit = np.append(self.recent_fit, [fit], axis=0)
		if self.recent_fit.shape[0] > self.n:
			self.recent_fit = np.delete(self.recent_fit, 0, 0)
		self.best_fit = np.average(self.recent_fit, axis=0)





def pipeline(img, objpoints, imgpoints, left_line, right_line, visualize=True):
	undistorted = cal_undistort(img, objpoints, imgpoints)
	threshed, color_binary = img_from_thresh(undistorted)
	binaries = [img, threshed, color_binary]
	titles = ['Original Image', 'Image after applying color and gradient thresholds', 'Color binary from each threshold']
	# showImgs(binaries, titles)

	img_size = (undistorted.shape[1], undistorted.shape[0])

	x_offset = 320 # offset for dst points
	y_offset = 0

	src = [[592, 450], [687, 450], [1120, img_size[1]], [198, img_size[1]]]
	dst = np.float32([[x_offset, y_offset], [img_size[0]-x_offset, y_offset], [img_size[0]-x_offset, img_size[1]-y_offset], [x_offset, img_size[1]-y_offset]])
	
	# print("src\n", src, "\ndst\n", dst)

	warped = warp(threshed, np.array(src, np.float32), dst, img_size)
	Minv = cv2.getPerspectiveTransform(dst, np.array(src, np.float32))

	# src_mask = region_of_interest(undistorted, src)
	# dst_mask = region_of_interest(warped, dst)
	# plt.imshow(warped)
	# plt.show()
	# showImgs([src_mask, dst_mask], ["original", "warped images"])

	left_fit = None
	right_fit = None
	left_fitx = None
	right_fitx = None

	if left_line.detected == False:
		left_fit, right_fit, left_fitx, right_fitx, ploty = find_lanes(warped, visualize=visualize)
	else:
		left_fit, right_fit, left_fitx, right_fitx, ploty = find_next_lanes(warped, left_line.best_fit, right_line.best_fit, visualize=visualize)

	left_line.set_x_fit(left_fitx)
	left_line.set_fit(left_fit)
	right_line.set_x_fit(right_fitx)
	right_line.set_fit(right_fit)
	left_line.detected = True
	right_line.detected = True

	# calculate the size of the perspective transform destination window
	dst_frame_x = warped.shape[1] - 2 * x_offset
	dst_frame_y = warped.shape[0] - 2 * y_offset
	# use the window size to calcuate the curvature of the lane in real life.
	left_curverad, right_curverad = curvature(dst_frame_x, dst_frame_y, left_fitx, right_fitx, ploty)
	print(left_curverad, "m", right_curverad, "m")

	output = draw_lane(undistorted, warped, left_fitx, right_fitx, ploty, Minv, visualize=visualize)

	return output

left_line = Line()
right_line = Line()

def process_video_img(img):
	return pipeline(img, objpoints, imgpoints, left_line, right_line, visualize=True)


process_video_img(imgs[1])
# for img in imgs:
# 	pipeline(img, objpoints, imgpoints, Line(), Line())

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

# white_output = 'project_output.mp4'
# clip1 = VideoFileClip('project_video.mp4')
# # prev_lines = []
# # smooth = True
# white_clip = clip1.fl_image(process_video_img) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

print("done.")
# input("press")
