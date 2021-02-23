#!/usr/bin/env python
# coding: utf-8

# In[10]:


# !pip install google-cloud-vision
# !pip install --upgrade google-cloud-vision


# In[11]:


from google.cloud import vision_v1
# from google.cloud.vision import types
from google.cloud import vision
import os
from google.cloud.vision import ImageAnnotatorClient


# In[12]:


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ocrexample-305213-082c9c1c4a3f.json"
# GOOGLE_APPLICATION_CREDENTIALS="C:\Users\username\Downloads\my-key.json"


# In[13]:


def detect_text_local(file):
    ImageAnnotatorClient()
    with open(file,'rb') as image_file:
        content = image_file.read()
    image =    vision_v1.types.Image(content = content)
    response = ImageAnnotatorClient(image=image)
    texts = response
    print("texts:")
#     for text in texts:
#         print('\n"{}'.format(text.description))
#         vertices = (['({},{})'.format(vertex.x,vertex.y)
#                     for vertex in text.bounding_poly.verices])
#         print()
detect_text_local("./car_park.jpg")


# In[ ]:




