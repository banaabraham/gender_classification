import matplotlib.patches as patches
import cv2
from face_detect import *
import numpy as np
import matplotlib.pyplot as plt
import pandas
from keras import layers
from keras import models
from keras.optimizers import RMSprop
import scipy
import os


direktori = "directory that contains man and woman folder"


def imageToArrayData(Dir,c_code):
    label = []
    imageArray = []
    train_names = []
    for file in os.listdir(Dir):
        if file.endswith(".jpg"):
            train_names.append(file)
    for name in train_names:
        try:
            img = crop_face(Dir+"\\"+name)
            y = cv2.resize(img,(150,150)) 
            y = y /255
            label.append(c_code)
            imageArray.append(y)
        except:
            pass
    return np.array(imageArray),label

label = os.listdir(direktori)

train = []
target = []

for i,l in enumerate(label):
    tempdir = direktori+"\\"+l
    tempres = imageToArrayData(tempdir,i)
    train.append(tempres[0])
    target.append(tempres[1])

train = np.vstack(train)
target = np.hstack(target)

"""
4 layer CNN
"""

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam")
model.fit(train,target,epochs=50,batch_size=10,validation_split=2)

model.save("gender_class.h5")
