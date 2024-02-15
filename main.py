import cv2 as cv 
import numpy as np
import imutils as imt 

video = cv.VideoCapture('video.mp4')
fgbg = cv.createBackgroundSubtractorMOG2()
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
quantity_car = 0

while True:
    ret, frame = video.read()
    frame = imt.resize(frame,width=800)
    if not ret: break

    area = np.array([[0,450], [frame.shape[1] -10,450],[frame.shape[1]-10,300],[0,300]])


    imgauX = np.zeros(shape=(frame.shape[:2]),dtype=np.uint8)
    imgauX =cv.drawContours(imgauX,[area],-1 ,(255),-1)

    cv.drawContours(frame, [area], -1, (0, 255, 0), 2)
    cv.line(frame,(790,400),(2,400),(0,255,255),1)
    imgArea = cv.bitwise_and(frame,frame,mask=imgauX)
    fgMask = fgbg.apply(imgArea)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)
    fgMask = cv.dilate(fgMask, None, iterations=5)
    cnts = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    for c in cnts:
        if cv.contourArea(c) >= 4000 : 
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(frame,(x,y),(x+w, y+h),(0,255,0))
            if 390 < (y + h)< 400:
                quantity_car = quantity_car + 1
                cv.line(frame,(790,400),(2,400),(0,255,0),2)

    cv.rectangle(frame, (frame.shape[1]-70, 215), (frame.shape[1]-5, 270), (0, 255, 0), 2)    
    cv.putText(frame, str(quantity_car), (frame.shape[1]-55, 250), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),2)
    cv.imshow('Frame', frame)
    if cv.waitKey(5) & 0xFF == ord('s'):
        break

video.release()
cv.destroyAllWindows()