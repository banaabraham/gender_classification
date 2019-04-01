# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
from keras import models
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture("C:\\Users\\lenovo\\Downloads\\Video\\habib.mp4")
model = models.load_model("gender_classs.h5")
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
    
	# Capture frame-by-frame
    ret,frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
        )
    

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    try:
        c = frame[y:y+h, x:x+w]
        c = cv2.resize(c,(150,150))
        c = c.reshape((1,150,150,3))
        siap = model.predict(c)
        if siap>0.9:
            cv2.putText(frame,"WOMAN",(x,y),font,0.5,(0,255,0),2,cv2.LINE_AA)
        elif siap<=0.9:
            cv2.putText(frame,"MAN",(x,y),font,0.5,(0,0,255),2,cv2.LINE_AA)
    except:
        continue
	# Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
