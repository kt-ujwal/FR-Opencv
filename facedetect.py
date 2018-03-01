import cv2
import numpy as np
def detect():
   face_cascade =cv2.CascadeClassifier('/home/kt_ujwal/cascades/haarcascade_frontalface_default.xml')
   eye_cascade = cv2.CascadeClassifier('/home/kt_ujwal/cascades/haarcascade_eye.xml')
   camera=cv2.VideoCapture(0)
   while(True):
        ret,frame=camera.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            img= cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            eyes=eye_cascade.detectMultiScale(roi_gray,1.03,5,0,(40,40))
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow("MyCamera",frame)
        if cv2.waitKey(100) & 0xff==ord("q"):
           break
   camera.release()
   cv2.destroyWindow("MyCamera")
if __name__ =="__main__":
   detect()
         
