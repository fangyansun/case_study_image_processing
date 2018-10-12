import cv2

class FaceDetector:
	# define where the Cascade classifier resides
	def __init__(self, faceCascadePath):
		self.faceCascade = cv2.CascadeClassifier(faceCascadePath)


	# scaleFactor means how much the image size is reduced at each image scale. This value is used to create the scale pyramid in order to detect faces at multiple scales in the image
	# minNeighbors controls how many rectangles need to be detected for the window to be labeled a face
	# bounding boxes smaller than this size are ignored	
	def detect(self, image, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30)):
		rects = self.faceCascade.detectMultiScale(image, scaleFactor = scaleFactor, minNeighbors = minNeighbors, minSize = minSize, flags = cv2.CASCADE_SCALE_IMAGE)
		return rects