# -*- coding: utf-8 -*-
"""OPENCV_Face Detection_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eqYcRUK_IDO0ky__oKmxBK6N-so5e_OM
"""

import numpy as np
import pandas as pd
import cv2 
from google.colab.patches import cv2_imshow # for image display
# from skimage import io
# from PIL import Image 
# import matplotlib.pylab as plt

face_cascade = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')

face_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

image = cv2.imread('/content/drive/MyDrive/OPENCV-IMAGES/DEVI.jpg')

image.shape

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2_imshow(image)

face = face_cascade1.detectMultiScale(gray,1.1,4,minSize = (20,20))

for (x,y,w,h) in face:
  cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

cv2_imshow(image)
cv2.waitKey(5)