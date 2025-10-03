# USAGE
#python color_tracking.py
#python color_tracking.py --video color_tracking/iphonecase.mov
#python color_tracking.py --color red

# import the necessary packages
import numpy as np	
import argparse
import time
import cv2
import os


# TODO: revise this function to determine green and red
def color_range(color):
	if color == 'blue':
		# define the upper and lower boundaries for a color
		# to be considered "blue"
		blueLower = np.array([100, 67, 0], dtype="uint8")
		blueUpper = np.array([255, 128, 50], dtype="uint8")
		return (blueLower, blueUpper)
	elif color == 'green':
		# TODO: define green boundaries
		greenLower = np.array([0, 100, 0], dtype="uint8")
		greenUpper = np.array([50, 255, 50], dtype="uint8")
		return (greenLower, greenUpper)
	elif color == 'red':
		# TODO: define red boundaries
		redLower = np.array([0, 0, 100], dtype="uint8")
		redUpper = np.array([50, 50, 255], dtype="uint8")
		return (redLower, redUpper)
	else:
		# Default to blue if color not recognized
		blueLower = np.array([100, 67, 0], dtype="uint8")
		blueUpper = np.array([255, 128, 50], dtype="uint8")
		return (blueLower, blueUpper)


if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", required=False, help="path to the (optional) video file") #ap.add_argument("-c", "--color", required=False, default="blue", help="color option is blue, green, or red (default is blue)")
	ap.add_argument("-c", "--color", required=False, default="red", help="color option is blue, green, or red (default is blue)")
	args = vars(ap.parse_args())

	# load the recorded video via the provided path,
	# otherwise 0 (meaning live video stream)
	video = args["video"] if args["video"] is not None else 0 

	# Check if video file exists when specified
	if args["video"] is not None and not os.path.exists(args["video"]):
		print(f"Error: Video file '{args['video']}' not found!")
		exit()

	camera = cv2.VideoCapture(video)
	
	# Check if video capture opened successfully
	if not camera.isOpened():
		print(f"Error: Could not open video source: {video}")
		exit()

	# keep looping
	while True:
		# grab the current frame
		(grabbed, frame) = camera.read()

		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break

		# Check if frame is valid
		if frame is None:
			continue

		# determine which pixels fall within the color boundaries
		# and then blur the binary image
		colorLower, colorUpper = color_range(color=args["color"])
		color = cv2.inRange(frame, colorLower, colorUpper)
		color = cv2.GaussianBlur(color, (3, 3), 0)

		# find contours in the image
		cnts, _ = cv2.findContours(color.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)

		# check to see if any contours were found
		if len(cnts) > 0:
			# sort the contours and find the largest one -- we
			# will assume this contour correspondes to the area
			# of my phone
			cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

			# Only draw if contour is large enough to avoid noise
			if cv2.contourArea(cnt) > 100:
				# compute the (rotated) bounding box around then
				# contour and then draw it		
				rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
				cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

		# show the frame and the binary image
		cv2.imshow("Tracking", frame)
		cv2.imshow("Binary", color)

		# if your machine is fast, it may display the frames in
		# what appears to be 'fast forward' since more than 32
		# frames per second are being displayed -- a simple hack
		# is just to sleep for a tiny bit in between frames;
		# however, if your computer is slow, you probably want to
		# comment out this line
		time.sleep(0.025)

		# if the 'q' key is pressed, stop the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	# cleanup the camera and close any open windows
	camera.release()
	cv2.destroyAllWindows()