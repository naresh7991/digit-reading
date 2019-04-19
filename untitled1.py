#importing the module

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.externals import joblib
from tensorflow.keras.models import model_from_json
import os
import cv2
import numpy as np
#load model from disk
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#creating the list of categories
listt = ['0','1','2','3','4','5','6','7','8','9']
#reading the image
im = cv2.imread("photo_1.jpg")
#image preprocessing
im_gray= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
im2,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#calculating the rectangle around the dgit
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
x = []
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    roi  = roi/255.0
    roi =  np.expand_dims(roi, axis=0)
    #predicting the image
    text = loaded_model.predict(roi)
    x.append(listt[text.argmax()])
    cv2.putText(im, listt[text.argmax()], (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    
cv2.imshow("Resulting Image with Rectangular ROIs", im)
#imagw close after 6 sec
cv2.waitKey(6000)
cv2.destroyAllWindows()
#printing the number
x = "".join(x)
print(x)

