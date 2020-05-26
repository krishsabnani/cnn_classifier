#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model


# In[3]:


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'


# In[4]:


##preprocessing
# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')


# In[5]:


def newLayer():
    model.add(Convolution2D(128, (3, 3), input_shape=(img_width, img_height,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


# In[6]:



model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[8]:


model.add(Flatten())
model.add(Dense(128,activation='relu'))

model.add(Dense(1,activation='sigmoid'))


# In[9]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[10]:


epochs = 5
train_samples = 2048
validation_samples = 832


# In[11]:


result = model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples// batch_size,)
#About 60 seconds an epoch when using CPU


# In[12]:


result.history


# In[13]:


model.save('models/model_basic_cnn_30_epochs.h5')


# In[14]:


model.save_weights('models/basic_cnn_30_epochs.h5')


# In[19]:


acc = result.history['val_accuracy'][4]*100


# In[21]:


file1=open("accuracy.txt","w")
file1.write(str(result.history['val_accuracy'][4]*100))
file1.close()


# In[ ]:




