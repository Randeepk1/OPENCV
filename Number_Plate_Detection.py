#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


framewidth = 550
frameheight = 440
minArea =500
color = (255,0,255)
num_plate = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')


# In[3]:


cap = cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(3,frameheight)
cap.set(10,100)
count =0
while True:
    rect,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    num_plates = num_plate.detectMultiScale(gray,1.1,10)
    for(x,y,w,h) in num_plates:
        area = w*h
        if area > minArea:    
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,"Number plate",(x,y-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
          
            imgRoi = frame[y:y+h,x:x+w]
            
            cv2.imshow("ROI",imgRoi)
            
    cv2.imshow("Result",frame)
      
    if cv2.waitKey(500) & 0xFF == ord('q'):
        cv2.imwrite("Numerplate/Scanned/Numplate"+str(count)+".jpg",imgRoi)
        cv2.rectangle(frame,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(frame,"scan saved",(150,265),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        cv2.imshow("Result",frame)
          
        count +=1
    
cap.release()
cv2.destroyAllWindows()        


# In[ ]:




