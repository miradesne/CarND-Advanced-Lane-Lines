import numpy as np
import cv2
import matplotlib.pyplot as plt

# y_eval is where we want to define the curvature.
def curvature(x_pix, y_pix, leftx, rightx, ploty):

	ym_per_pix = float(30/y_pix) # meters per pixel in y dimension
	xm_per_pix = 3.7/x_pix # meters per pixel in x dimension
	y_eval = y_pix # measure at the bottom of the view
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	# Now our radius of curvature is in meters

	# Calculate the distance between car center and lane center
	# Get the left and right lane x position on the bottom of the image
	left_start = leftx[-1] 
	right_start = rightx[-1]

	# Calculate lane center
	lane_center = left_start + (right_start - left_start) / 2
	car_center = x_pix / 2

	# Transform to real world length
	offset = abs(lane_center - car_center) * xm_per_pix
	return left_curverad, right_curverad, offset

def find_lanes(binary_warped, margin=100, minpix=50, nwindows=9, visualize=True):
	# Take a histogram of the bottom half of the image
	half = int(binary_warped.shape[0]/2)
	histogram = np.sum(binary_warped[half:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Set the width of the windows +/- margin
	# Set minimum number of pixels found to recenter window
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)


	# visualization:
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	if visualize:
		plt.imshow(out_img)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.show()

	return left_fit, right_fit, left_fitx, right_fitx, ploty

def find_next_lanes(binary_warped, left_fit, right_fit, margin=100, visualize=True):
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# visualization:
	if visualize:
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		window_img = np.zeros_like(out_img)
		# Color in left and right line pixels
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
		plt.imshow(result)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.show()

	return left_fit, right_fit, left_fitx, right_fitx, ploty