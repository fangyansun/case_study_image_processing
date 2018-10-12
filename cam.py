from __future__ import print_function
from facedetector import FaceDetector
import imutils
import argparse
import cv2

# video file is not necessary. We use it in case our webcam is not working.
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True, help = "Path to where the face cascade resides")
ap.add_argument("-v", "--video", help = "Path to (optional) video file")
args = vars(ap.parse_args())

if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])


# the loop stops when the video has reached its end and there are no more frames, or the user stops the execution of the script
while True:
	# grabbed is a boolean indicating whether reading the frame was successful
	(grabbed, frame) = camera.read()
	if args.get("video") and not grabbed:
		break

	frame = imutils.resize(frame, width = 300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# we need to tune the parameters so as to make sure that it works well
	fd = FaceDetector(args["face"])
	faceRects = fd.detect(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30))
	frameClone = frame.copy()

	# print the number of faces detected
	print("I found {} face(s)".format(len(faceRects)))

	# draw a bounding box around the detected face
	for (x, y, w, h) in faceRects:
		cv2.rectangle(frameClone, (x,y), (x+w, y+h), (0,255,0), 2)

	cv2.imshow("Faces", frameClone)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
