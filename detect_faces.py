from __future__ import print_function
from facedetector import FaceDetector
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True, help = "Path to where the face cascade resides")
ap.add_argument("-i", "--image", required = True, help = "Path to where the image file resides")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# we need to tune the parameters so as to make sure that it works well
fd = FaceDetector(args["face"])
faceRects = fd.detect(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30))

# print the number of faces detected
print("I found {} face(s)".format(len(faceRects)))

# draw a bounding box around the detected face
for (x, y, w, h) in faceRects:
	cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow("Faces", image)
cv2.waitKey(0)