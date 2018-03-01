import cv2
import numpy as np
import os
import sys

def generate():
   face_cascade =cv2.CascadeClassifier('/home/kt_ujwal/cascades/haarcascade_frontalface_default.xml')
   eye_cascade = cv2.CascadeClassifier('/home/kt_ujwal/cascades/haarcascade_eye.xml')
   camera=cv2.VideoCapture(0)
   fps = 50 # no. of frames per second
   numFramesRemaining = fps*6
   count=0
   while(True) and numFramesRemaining > 0:
        ret, frame=camera.read()
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            img= cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            eyes=eye_cascade.detectMultiScale(roi_gray,1.03,5,0,(40,40))
            for(ex,ey,ew,eh) in eyes:
               cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 

            f= cv2.resize(gray[y:y+h,x:x+h], (112,92))
  
            cv2.imwrite('/home/kt_ujwal/test_faces/ujwal/%s.pgm' %str(count),f)
            count+=1
            numFramesRemaining -=1#closing condition
        # 300 test faces of you are ready   
        cv2.imshow("MyCamera",frame)
        if cv2.waitKey(100) & 0xff == ord('q'):
           break
   camera.release()
   cv2.destroyAllWindows()

if __name__ =="__main__":
   generate() 

def read_images(path, sz=None):

    c= 0
    X,y= [],[]
    for dirname,dirnames,filenames in os.walk('test_faces'):
       for subdirname in dirnames:
         subject_path =os.path.join(dirname,subdirname)
         for filename in os.listdir(subject_path):
             try:
                 if(filename== ".directory"):
                    continue
                 filepath = os.path.join(subject_path,filename)
                 im = cv2.imread(os.path.join(subject_path,filename),cv2.IMREAD_GRAYSCALE)


                #resize to given size (if given) for proper recognition :D
 
                 if (sz is not None):
                     im= cv2.resize(im,(112,92))
                
                 X.append(np.asarray(im, dtype=np.uint8))
                 y.append(c)
             except IOError as e:
                    errno, strerror = e.args 
                    print ("I/O error({0}): {1}". format(errno,strerror))
             except:
                  print ("Unexpected error:", sys.exc_info()[0])
                  raise
        
         c=c+1
    return[X,y]   



                   
def face_rec():
    names =['s1','s2','s3','s4','s5','s6','s7','s8','s9','UJWAL']
    if len(sys.argv) <1:
      print ("Usage:facerec_demo.py </home/kt_ujwal/test_faces>[</home/kt_ujwal/test_faces/ujwal[]")
      sys.exit()

    [X,y] =read_images(sys.argv[0])
    y=np.asarray(y,dtype=np.int32)

    if len(sys.argv) == 2:
       out_dir = sys.argv[2]

    model = cv2.face.createEigenFaceRecognizer()
    model.train(np.asarray(X),np.asarray(y))
    camera=cv2.VideoCapture(0)
    face_cascade =cv2.CascadeClassifier('/home/kt_ujwal/cascades/haarcascade_frontalface_default.xml')
     
    fps = 50 # no. of frames per second
    numFramesRemaining = fps*3


    while(True) and numFramesRemaining > 0:
       read, frame =camera.read()
       frame=cv2.flip(frame,1,0) 
         
       faces=face_cascade.detectMultiScale(frame,1.3,5)
       numFramesRemaining -=1  
       for (x,y,w,h) in faces:
            
          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          roi =gray[x:x+w,y:y+h]
          try:
           roi =cv2.resize(roi,(112,92),interpolation=cv2.INTER_LINEAR) 
           
           params = model.predict(roi)
           print ("Label : %s, surety: %.2f"%(params[0],params[1]))
           cv2.putText(frame,names[params[0]],(x,y -20),cv2.FONT_HERSHEY_DUPLEX,1,255,10)
              
             
          except:
            continue
       cv2.imshow("MyCamera",frame)
       if cv2.waitKey(30) & 0xff == ord("q"):
          break
  
    cv2.destroyAllWindows()

if __name__ =="__main__":
   face_rec()



