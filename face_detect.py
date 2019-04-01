import cv2
import matplotlib.pyplot as plt
import os

def detect(image):
    cascPath = "haarcascade_frontalface_default.xml"
    
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(faces)!=0:
        result = []
        for i in faces.tolist():
            result.append(i)
        return result[0]
    else:
        return []


def crop_face(img_path):
    image = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
    faces = detect(image)
    
    if len(faces)==0:
        return image
    else:
        x, y, w, h = faces
        crop_img = image[y:y+h, x:x+w]
        return crop_img
