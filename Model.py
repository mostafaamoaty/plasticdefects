import numpy as np
from sklearn.metrics import  accuracy_score
import keras
import keras.backend as K
from keras.layers import Dense, Dropout, Flatten,Input, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.applications import vgg16
from keras.utils import np_utils
from keras import optimizers
import random
import os
from keras.applications import VGG16
import cv2
from google.colab import files
from keras import models
from keras.models import model_from_json


training_imgs =[]
training_label =[]
testing_imgs =[]
testing_label =[]
dim =200

training_imgs=np.load("/content/drive/My Drive/Dataset/all_binary_training_imgs.npy")
training_label=np.load("/content/drive/My Drive/Dataset/all_binary_training_label.npy")
testing_imgs=np.load("/content/drive/My Drive/Dataset/all_binary_testing_imgs.npy")
testing_label=np.load("/content/drive/My Drive/Dataset/all_binary_testing_label.npy")

#Get back the convolutional part of a VGG network trained 
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

# change trainable bool parameter to false if need to remove layers
#for layer in model_vgg16_conv.layers:
    #layer.trainable = False
  
#Create your own input format (here dim*dim*3)
input = Input(shape=(dim,dim,3))

#Use The Generated Convolution Layers Model 
output_vgg16_conv = model_vgg16_conv(input)


#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096 ,activation='relu', name='fc1')(x)
#x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
#x = Dropout(0.5)(x)
x = Dense(2, activation='softmax', name='predictions')(x)

#Create our model 
my_model = Model(input=input, output=x)

#Select Loss Function & Optimizer Learning rate
my_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.06, clipnorm=0.5),
              metrics=['accuracy'])

#convert Labels list to the specific shape to fit with model requirements
target = keras.utils.to_categorical(training_label)


#start fitting the training data with labels in our model
hist = my_model.fit(np.array(training_imgs), np.array(target), validation_split=0.20, epochs=20,batch_size=32)

#plot The Relation Between Accurcey  in Taining & Validation Phases
metric =  'acc' 
plt.plot(hist.epoch, hist.history[metric], label='train')
plt.plot(hist.epoch, hist.history['val_{}'.format(metric)], label='val')
plt.xlabel('epoch')
plt.ylabel(metric)
plt.legend()
plt.show()

#plot The Relation Between Loss Values  in Taining & Validation Phases
metric1 =  'loss' 
plt.plot(hist.epoch, hist.history[metric1], label='train')
plt.plot(hist.epoch, hist.history['val_{}'.format(metric1)], label='val')
plt.xlabel('epoch')
plt.ylabel(metric1)
plt.legend()
plt.show()
########################  Saving  Training Model ##########################
# Save the Model Architecture as Json File
with open('/content/drive/My Drive/Dataset/model_architecture_V7.json', 'w') as f:
    f.write(my_model.to_json())
#Save The Model Weights as H5 File
my_model.save_weights("/content/drive/My Drive/Dataset/Model_Weights_V7.h5")
