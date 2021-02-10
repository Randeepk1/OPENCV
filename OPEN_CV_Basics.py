#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# # IMAGE_READING

# In[2]:


img = cv2.imread('Randeepk.jpg')


# cv2.imshow("Randeep",img)  # we can see orginal image
# cv2.waitKey(10)

# # showing image using matplotlib

# In[3]:


plt.imshow(img)  # grayscale_image


# # video capture

# In[4]:


# cap =  cv2.VideoCapture(0) # 0 means system cam,1 means another camara,also u can pass video path
# cap.set(3,640) #frame width u can change 640 
# cap.set(4,48)# frame height
# cap.set(10,100) # for brightess
# while True:
#     rect,frame = cap.read() # capturing camra and storing in frame variable,rect giving Bolean values(TRUE,or FALS)
#     cv2.imshow("video",frame)
#     if cv2.waitKey(0)& 0xFF == ord('q'):
#         break


# # canny image

# In[5]:


img1 = cv2.imread('Randeepk.jpg')
canny = cv2.Canny(img,100,100)
cv2.imshow("randeep",canny)
cv2.waitKey(0)


# # GaussianBlur

# In[6]:


img1 = cv2.imread('Randeepk.jpg')
canny = cv2.GaussianBlur(img1,(7,7),0)
cv2.imshow("randeep",canny)
cv2.waitKey(0)


# # dilate

# In[7]:


kernal = np.ones((5,5),np.uint8)
img1 = cv2.imread('Randeepk.jpg')
canny = cv2.dilate(canny,kernal)
cv2.imshow("randeep",canny)
cv2.waitKey(0)


# In[8]:


cv2.erode(canny,kernal)


# # Resize 

# In[9]:


img2 = cv2.imread('Randeepk.jpg')
img2.shape
img_3 = cv2.resize(img2,(1000,1000),3)
cv2.imshow("randeep",img_3)
cv2.waitKey(0)


# # Image Crop

# In[10]:


img_crop = img_3[0:500,200:700]
img_crop
cv2.imshow("randeep",img_crop)
cv2.waitKey(0)


# # Black color bakground or image

# In[11]:


img_black = np.zeros((512,512))
img_black
cv2.imshow("randeep",img_black)
cv2.waitKey(0)


# # Giving Blue color on black image

# In[12]:


img_black1 = np.zeros((512,512,3),np.uint8)
img_black1
img_black1[:]=255,0,0 
cv2.imshow("randeep",img_black1)
cv2.waitKey(0)


# # giving color in specific part

# In[13]:


img_black2 = np.zeros((512,512,3),np.uint8)
img_black2
img_black2[100:300,100:300]=255,0,0 
cv2.imshow("randeep",img_black2)
cv2.waitKey(0)


# # Line

# In[14]:


img_black3 = np.zeros((512,512,3),np.uint8)
img_black3
cv2.line(img_black3,(0,0),(300,300),(0,255,0),3)
cv2.imshow("randeep",img_black3)
cv2.waitKey(0)


# # Rectangle

# In[15]:


img_black3 = np.zeros((512,512,3),np.uint8)
img_rec = cv2.rectangle(img_black3,(400,100),(200,200),(0,255,255),cv2.FILLED)
cv2.imshow("randeep",img_rec)
cv2.waitKey(0)


# # Circle

# In[16]:


img_black3 = np.zeros((512,512,3),np.uint8)
img_circ = cv2.circle(img_black3,(400,50),30,(0,255,255),cv2.FILLED)
cv2.imshow("randeep",img_circ)
cv2.waitKey(0)


# # Put_Text

# In[17]:


img_black3 = np.zeros((512,512,3),np.uint8)
text=cv2.putText(img_black3,"RANDEEP",(120,250),cv2.FONT_HERSHEY_COMPLEX,2,(201,100,200),3)
cv2.imshow("randeep",text)
cv2.waitKey(0)


# In[18]:


img5 = cv2.imread('Randeepk.jpg')
horizondal = np.hstack(img5)
cv2.imshow("randeep",horizondal)
cv2.waitKey(0)


# In[ ]:




