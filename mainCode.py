#import libarary.
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os
import shutil 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix

#show image.

img=mpimg.imread("../input/good-guysbad-guys-image-data-set/train/savory/0012.jpg",1)
plt.imshow(img)
plt.show()
img=mpimg.imread("../input/good-guysbad-guys-image-data-set/train/unsavory/0004.jpg",1)
plt.imshow(img)
plt.show()

#load images 
SIZE=128
train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
val_datagen  = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(
    "../input/good-guysbad-guys-image-data-set/train",
    target_size=(SIZE,SIZE),
    batch_size=32,
    class_mode="binary"
)
val_set=train_datagen.flow_from_directory(
    "../input/good-guysbad-guys-image-data-set/valid",
    target_size=(SIZE,SIZE),
    batch_size=32,
    class_mode="binary"
)
test_set=test_datagen.flow_from_directory(
    "../input/good-guysbad-guys-image-data-set/test",
    target_size=(SIZE,SIZE),
    batch_size=32,
    class_mode="binary"
)

print(test_set.classes)


#build our CNN model.
CNN = Sequential()

CNN.add(Convolution2D(32,(3,3),input_shape=(SIZE,SIZE,3),activation="relu"))

CNN.add(MaxPooling2D(pool_size=(2,2)))

CNN.add(Convolution2D(32,(3,3),activation="relu"))

CNN.add(MaxPooling2D(pool_size=(2,2)))

CNN.add(Flatten())

CNN.add(Dense(units=128,activation="relu"))

CNN.add(Dense(units=1,activation="sigmoid"))

CNN.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


print(CNN.summary())

#some important callbacks.
ModelCheckpoint=tf.keras.callbacks.ModelCheckpoint(
    "./models",
    monitor="loss_accuracy",
    mode="min",
    save_best_only=True
)
EarlyStopping=tf.keras.callbacks.EarlyStopping(
    monitor="loss",patience=3
)

#training model
epo=50
model=CNN.fit_generator(
    training_set,
    epochs=epo,
    validation_data=val_set,
    callbacks=[ModelCheckpoint,EarlyStopping]
)

#accuracy model.
print(CNN.evaluate(test_set))


#visualize results.
plt.plot(range(1,38+1),model.history['loss'],label="train loss")
plt.plot(range(1,38+1),model.history["val_loss"],label="val loss")
plt.legend()
plt.show()
plt.plot(range(1,38+1),model.history['accuracy'],label="train acc")
plt.plot(range(1,38+1),model.history["val_accuracy"],label="val acc")
plt.legend()
plt.show()