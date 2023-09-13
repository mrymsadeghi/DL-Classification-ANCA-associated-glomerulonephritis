import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.io as sio
import os

filepathBestmodel='Inception_Ourdataset_best_model.epoch{epoch:02d}-Loss{val_loss:.2f}-Accu{val_acc:.2f}.hdf5'
num_epochs=50
batch_size = 64
input_shape = (150,150)
num_finetuned_layers=0
train_dir = r'.\Added-Dataset1&2&3_nobg\train'
test_dir = r'.\Added-Dataset1&2&3_nobg\validation' 

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.20,  
    rotation_range = 20,
    shear_range=.15,
    horizontal_flip=True,vertical_flip=True,width_shift_range=0.15,
    height_shift_range=0.15,brightness_range=[0.80,1.20])


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator=train_datagen.flow_from_directory(train_dir,
                                            class_mode="categorical", 
                                            target_size=input_shape, 
                                            batch_size=batch_size)

validation_generator=test_datagen.flow_from_directory(test_dir,
                                            class_mode="categorical", 
                                            target_size=input_shape, 
                                            batch_size=batch_size)

###############################

#*********************#
from tensorflow.keras.layers import Flatten, Dense,Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
base_model=InceptionV3(include_top=False,input_shape=(150,150,3),weights='imagenet')

print(len(base_model.layers))

# fix the feature extraction part of the model
for layer in base_model.layers:
    layer.trainable=False

#let the last layers to be finetuned
for i_layer in range(len(base_model.layers)-30,len(base_model.layers)):
    base_model.layers[i_layer].trainable=True

x=Flatten()(base_model.output)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.2)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
#####################################

checkpoint = ModelCheckpoint(filepath=filepathBestmodel, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.samples // batch_size + 1,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size + 1,
                    epochs=num_epochs,                           
                    workers=8,             
                    max_queue_size=32,             
                    verbose=1, callbacks=callbacks)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#################################
