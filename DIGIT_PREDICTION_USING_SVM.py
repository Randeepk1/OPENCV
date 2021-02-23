#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from sklearn import datasets
from sklearn.svm import SVC
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


IMAGE_DIR = 'images'
TEST_IMAGE = 'five_new.jpg'


# In[3]:


digits = datasets.load_digits()
x,y = digits.data,digits.target


# In[4]:


print(x.shape)


# In[5]:


x


# In[6]:


y


# In[7]:


clf = SVC(gamma=0.001)
clf.fit(x,y)


# In[8]:


orginal_image = imread(os.path.join(TEST_IMAGE))


# In[12]:


img = resize(orginal_image,(8,8))
img = rescale_intensity(img,out_range=(0,16))


# In[13]:


x_test = [sum(pixel)/3 for row in img for pixel in row]
print("the predicted digit is {}".format(clf.predict([x_test])))


# In[17]:


plt.subplot(121);plt.title("original image");plt.imshow(orginal_image)
plt.subplot(122);plt.title("rescaled image");plt.imshow(img)
plt.show()
plt.imshow((img * 255).astype(np.uint8))


# In[ ]:





# In[ ]:




