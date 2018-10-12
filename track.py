from __future__ import print_function
import time
import argparse
import numpy as np
import cv2

# video file is not necessary. We use it in case our webcam is not working.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "Path to (optional) video file")
args = vars(ap.parse_args())

colorLower = np.array([100, 67, 0], dtype = "uint8")
colorUpper = np.array([255, 128, 50], dtype = "uint8")

camera = cv2.VideoCapture(args["video"])

# the loop stops when the video has reached its end and there are no more frames, or the user stops the execution of the script
while True:
	# grabbed is a boolean indicating whether reading the frame was successful
	(grabbed, frame) = camera.read()
	if not grabbed:
		break

	frame_new = cv2.inRange(frame, colorLower, colorUpper)
	frame_new = cv2.GaussianBlur(frame_new, (3,3),0)

	# we use cv2.findContours function to get the contour of the object
	(_, cnts, _) = cv2.findContours(frame_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# if we find any contour of the object, we take the biggest one	
	if len(cnts)>0:
		cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
		
		# draw the contour
		rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
		cv2.drawContours(frame, [rect], -1, (0,255,0), 2)

	# show images  --> video 
	cv2.imshow("Tracking", frame)
	cv2.imshow("framnew", frame_new)

	time.sleep(0.025)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break


camera.release()
cv2.destroyAllWindows()