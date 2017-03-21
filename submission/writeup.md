**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.png "Undistorted"
[image2]: ./output_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./project_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `get_image_points.py` and `calibration.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I passed in the number of corners in x and y direction, and used `cv2.findChessboardCorners()` to get the corner points.
After I get the points, I saved them in `calibration_points.p`

For the calibration and undistort part, I defined `cal_undistort()` in `calibration.py`. It takes in image points, object points, and the image. It returns a distorted image. It utilizes the `cv2.calibrateCamera()` function to get the distortion coefficients, and the `cv2.undistort()` function to do distortion correction on the images. 
I then load the output `objpoints` and `imgpoints` from `calibration_points.p` and used `cal_undistort()` to get the following result on a chess board image. 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
I used `cal_undistort()` in `calibration.py`, and `objpoints` and `imgpoints` from `calibration_points.p` to get the following undistorted image:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at `img_from_thresh()` in `thresh.py`).  

I transformed the color space to hsl, and use both h and s channels. I also use the x sobel oprator to get the horizontal gradient. With these two, I then combine them with the gradient direction and magnitude thresholds.

Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()` in `calibration.py`. The `warp()` function takes as inputs an image (`img`), image size (`img_size`) as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
x_offset = 320
y_offset = 0
src = [[592, 450],
		[687, 450], 
		[1120, img_size[1]], 
		[198, img_size[1]]]

dst = np.float32([
	[x_offset, y_offset], 
	[img_size[0]-x_offset, y_offset], 
	[img_size[0]-x_offset, img_size[1]-y_offset], 
	[x_offset, img_size[1]-y_offset]
])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 592, 450      | 320, 0        | 
| 687, 450      | 960, 0        |
| 1120, 720     | 960, 720      |
| 198, 720      | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the sliding window procedure from the lessons and defined a function called `find_lanes()` in `sliding_window.py`. It uses a histogram from the warped image, go from the middle of the image, and search for lanes indexes. After it gets all the coordinates of the lane lines, it uses `np.polyfit()` to fit a 2nd order polynomial line on the region.

Here's an output of the fit lines:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `curvature()` in my code in `sliding_window.py`. The function takes an input of a warped image, the x coordinates of the polynomial fitted lines, and returns the curvature in meters. 

I used the formula described in the lesson(R = (1+(2Ay+B)^2)^1.5 / |2A|), and time all the parameters with the pixel-to-meter ratio to get the real world length. The curvature measured is on the bottom of the image, where it's the closest to the car.

An example output of the curvatures is 404.518674424 m 287.199316844 m.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 40 through 60 in my code in `utilities.py` in the function `draw_lane()`.
I use an inverse matrix to warp the shape back to the original shape, and use `cv2.polyfill()` to fill up the lane area.
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One approach I did was to use the previous fitted lane lines as a starting point for the next line search. The function is called `find_next_lanes()` in `sliding_window.py`. I also use an average of most recent 5 lines instead of the exact last lines to produce a smoothier result and prevent error. It worked well on parts of the roads where the shades make detection very hard. 

However I think I can improve the thresholds to detect better lane lines. At some areas there are a lot of noices. One noice that makes it especially hard to detect is the edge of the road. It can have great gradient. In that case the algorithm may try to identify that as the lane lines instead of the real yellow lines exposed in the sun and have a much smaller contrast. I think we can use deep learning to extract the features out.  The model can use te features to identify if they are lane lines, or just some edges with great gradient.
