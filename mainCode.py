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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import cv2
from PIL import Image

#show image.

img=mpimg.imread("../input/good-guysbad-guys-image-data-set/train/savory/0012.jpg",1)
plt.imshow(img)
plt.show()
img=mpimg.imread("../input/good-guysbad-guys-image-data-set/train/unsavory/0004.jpg",1)
plt.imshow(img)
plt.show()

#load images 
SIZE=128
training_data = []
training_path = "../input/good-guysbad-guys-image-data-set/train"
categories = ["savory", "unsavory"]
for category in categories:  
    path = os.path.join(training_path,category)  
    class_num = categories.index(category)  
    
    for img in tqdm(os.listdir(path)): 
        img_array = cv2.imread(os.path.join(path,img) ,1) 
        new_array = cv2.resize(img_array, (SIZE, SIZE))
        training_data.append([new_array, class_num])  

from sklearn.utils import shuffle
X=[]
y=[]
for img,res in training_data:
    X.append(img)
    y.append(res)
X = np.array(X).reshape(-1,SIZE, SIZE, 3) 
y = np.array(y)
X, y = shuffle(X, y)


valid_data = []
valid_path = "../input/good-guysbad-guys-image-data-set/valid"
categories = ["savory", "unsavory"]
for category in categories:  
    path = os.path.join(valid_path,category)  
    class_num = categories.index(category)  
    
    for img in tqdm(os.listdir(path)): 
        img_array = cv2.imread(os.path.join(path,img) ,1) 
        new_array = cv2.resize(img_array, (SIZE, SIZE))
        valid_data.append([new_array, class_num])  

#from sklearn.utils import shuffle
X_val=[]
y_val=[]
for img,res in valid_data:
    X_val.append(img)
    y_val.append(res)
X_val = np.array(X_val).reshape(-1,SIZE, SIZE, 3) 
y_val = np.array(y_val)
X_val, y_val = shuffle(X_val, y_val)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , shuffle=True)



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
model=CNN.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=epo,
    validation_data=(X_val,y_val),
    callbacks=[ModelCheckpoint,EarlyStopping]
)

#accuracy model.
print(CNN.evaluate(X_test,y_test))


#visualize results.
plt.plot(range(1,len(model.history['loss'])+1),model.history['loss'],label="train loss")
plt.plot(range(1,len(model.history['val_loss'])+1),model.history["val_loss"],label="val loss")
plt.legend()
plt.show()
plt.plot(range(1,len(model.history['accuracy'])+1),model.history['accuracy'],label="train acc")
plt.plot(range(1,len(model.history['val_accuracy'])+1),model.history["val_accuracy"],label="val acc")
plt.legend()
plt.show()