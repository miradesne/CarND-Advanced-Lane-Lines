import matplotlib.pyplot as plt
import numpy as np
import cv2

def showImgs(imgs, titles):
	count = len(imgs)
	f, axes = plt.subplots(1, count, figsize=(24, 9))
	f.tight_layout
	for i in range(len(axes)):
		img = imgs[i]
		ax = axes[i]
		if len(img.shape) == 3:
			ax.imshow(img)
		else:
			ax.imshow(img, cmap ='gray')
		ax.set_title(titles[i], fontsize=12)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	return cv2.addWeighted(initial_img, α, img, β, λ)

def region_of_interest(img, vertices):
	if len(img.shape) == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		img = img * 255

	pts = np.array(vertices, np.int32)
	pts = pts.reshape((-1,1,2))
	masked_image = cv2.polylines(img,[pts],True,(255,0,0), thickness = 2)
	return masked_image


def draw_lane(image, warped, left_fitx, right_fitx, ploty, Minv, curvature, visualize=True):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

	font = cv2.FONT_HERSHEY_SIMPLEX
	result = cv2.putText(result,curvature,(50,100), font, 1.2,(255,0,0),3)

	if visualize:
		plt.imshow(result)
		plt.show()
	return result
