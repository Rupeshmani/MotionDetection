#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:22:33 2022

@author: rupeshmani
"""

import cv2   #image
import time  #delay
import imutils #resize

cam= cv2.VideoCapture(0)
time.sleep(1)
firstFrame,area=None,500

while True:
    check,img=cam.read() #read frame from video
    text="Normal"
    img= imutils.resize(img,width=500)
    grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianImg=cv2.GaussianBlur(grayImage,(21,21),0) #smoothing
    
    if firstFrame is None:
        firstFrame=gaussianImg
        continue
    
    imgDiff = cv2.absdiff(firstFrame,gaussianImg)

    threshImg = cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]

    threshImg = cv2.dilate(threshImg,None,iterations=2)

    cnts = cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    for i in cnts:
        if cv2.contourArea(i)<area:
            continue
        (x,y,w,h)=cv2.boundingRect(i)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        text="Moving Object Detected"
    print(text)

    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("cameraFeed",img)

    key=cv2.waitKey(1) & 0xFF

    if key==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()