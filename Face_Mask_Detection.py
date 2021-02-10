#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# In[12]:


img = cv2.imread('Randeepk.jpg')


# In[13]:


img.shape


# In[14]:


img[0]  


# In[15]:


plt.imshow(img)


# In[6]:


# while True:
#     cv2.imshow('Randeep',img)
#     if cv2.waitKey(0) == 10:
#         break
#     cv2.destroyAllWindows()


# In[7]:


har_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[8]:


har_data.detectMultiScale(img)


# In[9]:


while True:
    faces = har_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4) # x is xlabel,y is y label,w is width,h is height,255,0,255 is color of rgb,4 is width of label
    cv2.imshow('Randeep',img)
    if cv2.waitKey(2) == 13:
        break
cv2.destroyAllWindows()


# In[16]:


cap = cv2.VideoCapture(0)
data = []
while True:
    rect,frame = cap.read()
    if rect:
        faces = har_data.detectMultiScale(frame)
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),5)
            face = frame[y:y+h,x:x+w,:]
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<300:
                data.append(face)
        cv2.imshow('Randeep',frame)
        if cv2.waitKey(2) == 13 or len(data)>=200:
            break
cap.release()
cv2.destroyAllWindows()


# In[ ]:


np.save("With_Mask_new.npy",data)


# In[ ]:


# np.save("Without_Mask_new.npy",data)


# In[ ]:


plt.imshow(data[1])


# In[ ]:


with_mask = np.load('With_Mask.npy')
without_mask = np.load('Without_Mask.npy')


# In[ ]:


with_mask.shape
without_mask.shape
with_mask = with_mask.reshape(200,50*50*3)
without_mask = without_mask.reshape(200,50*50*3)
with_mask.shape,without_mask.shape
x = np.r_[with_mask,without_mask]


# In[ ]:


x.shape


# In[ ]:


y = np.zeros(x.shape[0])


# In[ ]:


y[200:] =1.0


# In[ ]:


name = {0:'Mask',1:'No Mask'}


# In[ ]:


y.shape


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


# In[ ]:


pca = PCA(n_components=3)


# In[ ]:


x_train = pca.fit_transform(x_train)


# In[ ]:


x_test = pca.transform(x_test)


# In[ ]:


x_test.shape


# In[ ]:


x_train.shape


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state =42)


# In[ ]:


sv = SVC()


# In[ ]:


sv.fit(x_train,y_train)


# In[ ]:


sv = SVC()


# In[ ]:


sv.fit(x_train,y_train)


# In[ ]:


y_pred = sv.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


img = cv2.imread('Randeepk.jpg')
har_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    rect,frame = cap.read()
    if rect:
        faces = har_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),5)
            face = frame[y:y+h,x:x+w,:]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
#             pca = pca.transform(face)
            pred = sv.predict(face)[0]
            n = name[int(pred)]
            cv2.putText(frame,n,(x,y),font,1,(244,255,250),2)
            print(n)
            
        cv2.imshow('Randeep',frame)
        if cv2.waitKey(2) == 13:
            break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




