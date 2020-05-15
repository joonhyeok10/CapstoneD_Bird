#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[2]:


# keras import
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


# In[3]:


import pickle
import scipy


# In[4]:


# image preprocessing 및 훈련 중 실시간 증가
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


# In[5]:


data = open('..\\Caps\\full_dataset.pkl', "rb")
X_train, Y_train, X_test, Y_test = pickle.load(data, encoding='latin1')


# In[6]:


datagen.fit(X_train)


# In[7]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', 
                 input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.summary()

# 모델 컴파일
model.compile(loss='categorical_crossentropy',
             metrics=['accuracy'],
             optimizer='adam')


# In[8]:


n_epochs = 50
batch_size = 128

# 모델 학습
history = model.fit_generator(
  datagen.flow(X_train, Y_train, batch_size=batch_size), 
  steps_per_epoch=len(X_train) / batch_size, 
  epochs = n_epochs, verbose=2, 
  validation_data=(X_test, Y_test))


# In[9]:


# saving the model
save_dir = "..\\Caps\\bird-classifier"
model_name = 'keras_birds.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# In[34]:


# 이미지 입력
img = scipy.ndimage.imread('..\\Caps\\negative_test_image\\zet.jpg', mode="RGB")

# 입력 이미지 전처리
img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
img.reshape(1,32,32,3).shape

# 모델 불러오기
birds_model = load_model('..\\Caps\\bird-classifier\\keras_birds.h5')

# 예측
prediction = birds_model.predict(img.reshape(1,32,32,3))

# 결과
is_bird = np.argmax(prediction[0]) == 1

if is_bird:
    print("It's a bird!")
else:
    print("It's not a bird!")

