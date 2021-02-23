#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


img = cv2.imread("lena.jpg")
color = ('blue','green','red')
for i,col  in enumerate(color): #enumarete give index with value in color tuple ex : color = ('blue','green','red'), (0,blue),(1,green),(2,red)  
    hister = cv2.calcHist([img],[i],None,[256],[0,256])   # 1st parameter img,second channelindex here index is blue is 0 channale,1 is green,2 is red # here passing in the list,3rd if you have  plot another histogram u can give here or give None(appling another histogram),4th RGB range or histogram size,5th color range    
    plt.plot(hister,color=col[0]) 
    plt.xlim(0,256)
plt.show()


# In[13]:


imge2 =cv2.imread("sachin1.jpg")
color = ['blue','green','red']
for i,col in enumerate(color):
    hist = cv2.calcHist([imge2],[1],None,[256],[0,256])
    plt.plot(hist,color=col[0])
    plt.xlim(0,256)
plt.show()


# In[19]:


img = cv2.imread("RANDEEP.jpg")
color = ('blue','green','red')
for i,col  in enumerate(color):
    hister = cv2.calcHist([img],[i],None,[256],[0,256])        
    plt.plot(hister,color=col[0])
    plt.xlim(0,256)
plt.show()


# In[20]:


img = cv2.imread("m2.jpg")
color = ('blue','green','red')
for i,col  in enumerate(color):
    hister = cv2.calcHist([img],[i],None,[256],[0,256])        
    plt.plot(hister,color=col[0])
    plt.xlim(0,256)
plt.show()


# In[21]:


img = cv2.imread("m1.jpg")
color = ('blue','green','red')
for i,col  in enumerate(color):
    hister = cv2.calcHist([img],[i],None,[256],[0,256])        
    plt.plot(hister,color=col[0])
    plt.xlim(0,256)
plt.show()


# In[24]:


img1 = cv2.imread("dog.jpg")
color = ['blue','green','red']
for i, col in enumerate(color):
    hist = cv2.calcHist([img1],[i],None,[256],[0,256])
    plt.plot(hist,color=col[0])
    plt.xlim(0,256)
plt.show()


# In[28]:


img2 = cv2.imread("dog1.jpg")
color = ('blue','green','red')
for i,col in enumerate(color):
    hist = cv2.calcHist([img2],[i],None,[256],[0,256])
    plt.plot(hist,color=col[0])
    plt.xlim(0,256)
plt.show()

