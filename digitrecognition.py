import pandas as pd
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps 
import os, ssl, time
X,y = fetch_openml(‘mnist_784’, version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = [‘0’, ’1’, ‘2’ , ’3’, ‘4’, ‘5’, ‘6’ , ‘7’ ,’8’ ,’9’]
nclasses = len(classes)
#to make code secure (ssl)
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)): ssl._create_default_https_context = ssl._create_unverified_context
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
X_train_scale = X_train/255
X_test_scale = X_test/255
clf = LogisticRegression(solver='saga',multi_class = 'multinomial').fit(X_train_scale,y_train)
y_pred = clf.predict(X_test_scale)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
cap = cv2.VideoCapture(0)
while(True):
    #try block runs the rest of the code even when there is an error
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height,width = gray.shape()
        upper_left = (int(width/2-56),int(height/2-56))
        bottom_right = (int(width/2-+56),int(height/2+56))
        cv2.reactangle(gray, upper_left, bottom_right,(0,255,0),2)
        #roi is Region of Interest from where it picks up the numbers
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        im_pil = Image.fromarray(roi)
        Image_bw = im.pil.convert("L")   
        #antialias to remove rough edges(smoothing)
        Image_bw_resized = Image_bw_resize(28,28),Image.antialias
        Image_bw_resize_inverted = PIL.ImageOps.invert(Image_bw_resize)
        pixel_filter = 20
        min_pixel = np.percentile(Image_bw_resize_inverted, pixel_filter)
        Image_bw_inverted_scaled = np.clip(Image_bw_resize_inverted-min_pixel,0,255)
        max_pixel = np.max(Image_bw_inverted_scaled)
        Image_bw_inverted_scaled = np.asarray( Image_bw_inverted_scale)/max_pixel
        test_sample = np.array(Image_bw_inverted_scale).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print(test_pred)
        cv2.imshow("Frame",gray)
        cv2.waitkey(1)& 0xFF == ord('q'):
        break
    cap.release()
    cv2.DestroyAllWindows()
