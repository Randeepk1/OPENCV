#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# In[2]:


with_mask = np.load('With_Mask.npy')
without_mask = np.load('Without_Mask.npy')


# In[3]:


with_mask.shape
without_mask.shape
with_mask = with_mask.reshape(200,50*50*3)
without_mask = without_mask.reshape(200,50*50*3)
with_mask.shape,without_mask.shape
x = np.r_[with_mask,without_mask] # np.r_ means concatinating raws


# In[4]:


x.shape


# In[5]:


y = np.zeros(x.shape[0])


# In[6]:


y[200:] =1.0


# In[7]:


name = {0:'Mask',1:'No Mask'}


# In[8]:


y.shape


# In[9]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


# In[10]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[11]:


pca = PCA(n_components=3)


# In[12]:


x_train = pca.fit_transform(x_train)


# In[13]:


x_test = pca.transform(x_test)


# In[14]:


x_test.shape


# In[15]:


x_train.shape


# In[16]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state =42)


# In[17]:


sv = SVC()


# In[18]:


sv.fit(x_train,y_train)


# In[19]:


y_pred = sv.predict(x_test)


# In[20]:


accuracy_score(y_test,y_pred)


# In[21]:


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




