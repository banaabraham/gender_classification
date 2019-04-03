import cv2
from keras import models
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture("C:\\Users\\lenovo\\Downloads\\Video\\saori.mp4")
model = models.load_model("gender_class.h5")
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

gender = "NOT DETERMINED"
frames = 0
accum_frame = []
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
            accum_frame.append(model.predict(c))
        except:
            continue
        
    #determine the gender every 10 frames based on the accum_frame mean
    if frames%10==0:
        siap = np.mean(accum_frame)
        if siap>0.5:
            gender = "MALE"
        elif siap<=0.5:
            gender = "FEMALE"
        else:
            gender = ""
            
    cv2.putText(frame,gender,(x,y),font,0.5,(0,255,0),2,cv2.LINE_AA)      
	# Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frames+=1

cap.release()
cv2.destroyAllWindows()
