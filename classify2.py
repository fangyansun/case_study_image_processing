from __future__ import print_function
from rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import numpy as np
import argparse
import glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True)
ap.add_argument("-m", "--masks", required = True)
args = vars(ap.parse_args())


imagePaths = sorted(glob.glob(args["images"] + "/*.png"))
maskPaths = sorted(glob.glob(args["masks"] + "/*.png"))

data = []
target = []

desc = RGBHistogram([8, 8, 8])


# get data, features and target in a list
for (imagePath, maskPath) in zip(imagePaths, maskPaths):
    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    features = desc.describe(image, mask)
    
    data.append(features)
    target.append(imagePath.split("_")[-2])
 
# to transform tartget to an integer forma
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

# to split data into train data and test data
(trainData, testData, trainTarget, testTarget) = train_test_split(data, target, test_size = 0.3, random_state = 42)

# we use random forest algo for classification        
model = RandomForestClassifier(n_estimators = 25, random_state = 84)
model.fit(trainData, trainTarget)

print(classification_report(testTarget, model.predict(testData), target_names = targetNames))

# to visualize results with some examples
for i in np.random.choice(np.arange(0, len(imagePaths)), 10):
    print(i)
    imagePath = imagePaths[i]
    maskPath = maskPaths[i]
    
    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    features = desc.describe(image, mask)
    
    flower = le.inverse_transform(model.predict([features]))[0]

    # show result
    result = str("I think this flower is a {}".format(flower.upper()))
    cv2.putText(image, str(imagePath), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, result, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)