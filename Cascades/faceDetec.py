import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_mcs_righteye_alt.xml')

img = cv2.imread('teste1.pgm')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    #crop_img = img[y: y + h, x: x + w]

#cv2.imshow('crop_img',crop_img)
cv2.imshow('mg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
