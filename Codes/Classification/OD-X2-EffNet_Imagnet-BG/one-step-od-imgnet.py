#*******************************importing necessary packages**************************
import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import scipy.io as sio
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras import Model
import efficientnet.tfkeras as efn
#*******************************Setting parameters************************************
input_shape = (240,240)
#The path of Best trained model
filepathBestmodel='OneStep_pathologydataset_Ourdataset_best_model.epoch{epoch:02d}-Loss{val_loss:.2f}-Accu{val_acc:.2f}.hdf5'

#Second phase parameters
second_num_epochs=50
second_num_finetuned_layers=00
second_batch_size = 16
second_train_dir = r'.\Added-Dataset1&2&3_bg\train'
second_val_dir = r'.\Added-Dataset1&2&3_bg\validation' 

second_train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.15,  
    rotation_range = 15,
    horizontal_flip=True,vertical_flip=True,width_shift_range=0.15,
    height_shift_range=0.15,brightness_range=[0.80,1.20])

second_val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator=second_train_datagen.flow_from_directory(second_train_dir,
                                            class_mode="categorical", 
                                            target_size=input_shape, 
                                            batch_size=second_batch_size)

validation_generator=second_val_datagen.flow_from_directory(second_val_dir,
                                            class_mode="categorical", 
                                            target_size=input_shape, 
                                            batch_size=second_batch_size)

#save train_datagen.class_indices on disk to use it later 
with open('train_generator_classindices.pkl', 'wb') as outp:
    pickle.dump(train_generator.class_indices, outp, pickle.HIGHEST_PROTOCOL)


#Load pretrained model from EfficeintNet
base_model = efn.EfficientNetB1(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
second_model = Model(inputs=base_model.input, outputs=predictions)

# fix the feature extraction part of the model
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False
print(len(base_model.layers))
for i_layer in range(len(base_model.layers)-second_num_finetuned_layers,len(base_model.layers)):
    base_model.layers[i_layer].trainable=True

second_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
second_model.summary()
#define check point for saving best model
checkpoint = ModelCheckpoint(filepath=filepathBestmodel, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

history = second_model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.samples // second_batch_size + 1,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // second_batch_size + 1,
                    epochs=second_num_epochs,                           
                    workers=8,             
                    max_queue_size=32,             
                    verbose=1, callbacks=callbacks)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#################################
second_model.save('finalmodel.h5')
