import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Input,Convolution2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from keras.utils import np_utils
import tensorflow

# images path
SUV_path = "./data/car_data/train/SUV/"
Sedan_path = "./data/car_data/train/sedan/"



# Data Creation
images = np.zeros((2757,100,100))
labels = np.zeros((2757,1))


suvs = os.listdir(SUV_path)
sedans = os.listdir(Sedan_path)

for ix,iy in enumerate(suvs):
    img = cv2.imread(SUV_path + iy, 0)
    img = cv2.resize(img, (100,100))
    images[ix] = img
    labels[ix] = 1


for ix,iy in enumerate(sedans):
    img = cv2.imread(Sedan_path + iy, 0)
    img = cv2.resize(img, (100,100))
    images[ix+len(suvs)] = img


images, labels = shuffle(images, labels, random_state=0)
y = np_utils.to_categorical(labels)
images = images/255.0
X = images.reshape(-1,100,100,1)
print (X.shape,y.shape)


## CNN Model
model = Sequential()
model.add(Convolution2D(16,(3,3),activation='relu', input_shape=(100,100,1)))
model.add(BatchNormalization())
model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(64,(5,5),activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(32, (5,5),activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(16,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(8, (3,3),activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2,activation='softmax'))


# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# model training
hist = model.fit(X,y,
                 epochs = 2,
                 shuffle = True,
                 batch_size = 128)


# saving models
model.save("cars_model.h5")


