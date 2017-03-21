import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def abs_sobel_thresh(img, orient='x', abs_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= abs_thresh[0]) & (scaled_sobel <= abs_thresh[1])] = 1

    # Return the result
    return binary_output

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
# image = mpimg.imread('bridge_shadow.jpg')

# Edit this function to create your own pipeline.
def img_from_thresh(img, h_thresh=(15, 30), s_thresh=(170, 255), sx_thresh=(20, 100)):

	img = np.copy(img)
	# Convert to HSV color space and separate the V channel
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	h_channel = hsv[:,:,0]
	l_channel = hsv[:,:,1]
	s_channel = hsv[:,:,2]
	# Sobel x
	sxbinary = abs_sobel_thresh(img, orient='x', abs_thresh=sx_thresh)

	# Gradient direction + magnitude
	d_binary = dir_threshold(img, thresh = (0.7, 1.3))
	mag_binary = mag_threshold(img, mag_thresh = (100, 255))

	# Threshold s and h channels
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	
	h_binary = np.zeros_like(h_channel)
	h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
	
	# Stack each channel
	# Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
	# be beneficial to replace this channel with something else.
	color_binary = np.dstack((mag_binary, sxbinary, s_binary))

	combined = np.zeros_like(sxbinary)

	# choose which threshed binaries we want to use
	gradient_and_color = ((d_binary == 1) & (mag_binary == 1)) | (sxbinary == 1) | ((s_binary == 1) & (h_binary == 1))
	
	# color_only = (s_binary == 1) & (h_binary == 1)
	combined[gradient_and_color] = 1
	return combined, color_binary
	
# result = img_from_thresh(image)

# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()

# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=40)