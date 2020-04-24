#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
 
import datetime
import pandas as pd
import os
import warnings

from PIL import Image
import numpy as np
from numpy import save, load

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization,     GlobalMaxPool2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Conv2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

import cv2

from tqdm import tqdm
from collections import Counter


# In[2]:


#K.set_floatx('float16')


# In[ ]:





# In[3]:


logging.basicConfig(level=logging.DEBUG)
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")


# In[4]:


def Get_Croped_image(img,bb_data):
    img_shape = img.shape
    x = bb_data[0]
    y = bb_data[1]
    w = bb_data[2]
    h = bb_data[3]
    crop_img = img[y:y+h, x:x+w]
    return crop_img


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    zoom_range=0.1)

def augment(im_array):
    im_array = datagen.random_transform(im_array)
    return im_array


def read_and_resize(dataFrame, input_shape=(224,224),aug=True):
    #filepath = image_data.path.values[0]
    images = []
    for index in dataFrame.index :
        im_cv = cv2.imread(dataFrame["path"][index])
        bb_data = (dataFrame["x"][index],dataFrame["y"][index],dataFrame["w"][index],dataFrame["h"][index])
        im_cv = Get_Croped_image(im_cv,bb_data)
        im_cv = cv2.resize(im_cv,input_shape)
        im_array = np.array(im_cv)
        im_cv = np.array(im_array / (np.max(im_array)+ 0.001))
        if aug:
            im_cv = augment(im_cv)
        images.append(im_cv)
    return images


# In[5]:



def gen(df, batch_size=4,input_shape=(64,64), aug=False):
    df = df.sample(frac=1)
    while True:
        for _, batch in enumerate([df[i:i+batch_size] for i in range(0,df.shape[0],batch_size)]):
            labels = np.array(batch.out_ages.values)
            labels = labels[..., np.newaxis]    
            images = np.array(read_and_resize(batch,aug=aug))
            yield images, labels


# In[ ]:





# In[6]:


def get_model(optimizer,n_classes=1):

#     base_model = keras.applications.nasnet.NASNetMobile(weights="./NASNet-mobile-no-top.h5", include_top=False)

#     #for layer in base_model.layers:
#     #    layer.trainable = False

#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(0.2)(x)
#     #x = Flatten()
#     x = Dense(1000, activation="relu")(x)
#     x = Dropout(0.2)(x)
#     x = Dense(750,activation="relu")(x)
#     x = Dense(350,activation="relu")(x)
#     x = Dense(100,activation="relu")(x)
#     x = Dropout(0.2)(x)
#     if n_classes == 1:
#         x = Dense(n_classes, activation="sigmoid")(x)
#     else:
#         x = Dense(n_classes, activation="softmax")(x)

#     base_model = Model(base_model.input, x, name="base_model")
#     if n_classes == 1:
#         base_model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=optimizer)
#     else:
#         base_model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer=optimizer)

    base_model = load_model("imdb_age_recog_weights.h5")

    return base_model


# In[7]:


if __name__ == "__main__":
    base_path = "./Dataset-copy/"

    dict_age = {'(0, 2)' : 0,
                '(3, 5)' : 1,
                '(6, 10)' : 2,
                '(11, 15)' : 3,
                '(16, 20)' : 4,
                '(21, 30)' : 5,
                '(31, 40)' : 6,
                '(41, 50)' : 7,
                '(51, 60)' : 8,
                '(61, 70)' : 9,
                '(71, 80)' : 10,
                 '(81, 90)' : 11,
                 '(91, 100)' : 12}

    bag = 3

    all_indexes = list(range(5))
    
    accuracies = []
    print("Reading train and test CSV files ")
    train_df = pd.read_csv("croped_filter_expanded_data.csv")
    #test_df = pd.read_csv("test_gender_filtered_data_with_path.csv")
    tr_tr, tr_val = train_test_split(train_df, test_size=0.0005, random_state=100)
    tr_unique_ages = tr_tr['out_ages'].unique()
    tr_unique_ages.sort()
    #print("Unique ages are: ",val_unique_ages)
    print("Reading Done.")
    cnt_ave = 0
    predictions = 0
#     print("Extracting test labels and test images from files")
#     test_images = load("imdb_test_images.npy")
#     test_labels = load("imdb_test_labels.npy")
#     print("Extracting Done.")
    #tr_tr, tr_val = train_test_split(train_df, test_size=0.1,random_state = 100)
    file_path = "imdb_age_recog_weights.h5"
    
    print("Generating callback_list")
    
#     log_dir="./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    #early = EarlyStopping(monitor="val_acc", mode="max", patience=5)

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss",
                                          mode="min", 
                                          factor=0.1,
                                          #cooldown=0,
                                          patience=7,
                                          verbose=1,
                                          min_lr=0.0000001)

    tensorboard = TensorBoard(log_dir='./logs/imdb_age_recog')

    callbacks_list = [checkpoint,
                      reduce_on_plateau,
                      tensorboard
                      #tensorboard_callback
                      #early
                     ]  # early
    
    print("Done Generating callbacklist.")
    print("generating Model")
    optimizer = Adam(lr=0.0001)
    model = get_model( optimizer,n_classes=99)
    print("Done generating model")
    


# In[8]:


print("Running Fit_generator")
batch_size = 32
input_shape=(64,64)
model.fit_generator(gen(tr_tr,batch_size=batch_size, aug=True), 
                        validation_data=gen(tr_val), 
                        epochs=200, 
                        verbose=1, 
                        #workers=4,
                        callbacks=callbacks_list,
                        steps_per_epoch=int(len(tr_tr)/batch_size),#int(10740.75), 
                        validation_steps=len(tr_val))
                        #validation_data=((test_images), test_labels)
                        #use_multiprocessing=True)
    #model.save(file_path)
print("Trained Model saved  to disk")


# ##model.save("imdb_NAS_mobile_save.h5")

# In[ ]:




