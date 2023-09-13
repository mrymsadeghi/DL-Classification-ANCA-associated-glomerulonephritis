import os, sys
import scipy.io as sio
import os
import tensorflow as tf
import tensorflow
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras import Model
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
#
input_shape = (150,150)
test_dir=r'.\TestDataset_bg'
###################
with open('train_generator_classindices.pkl', 'rb') as inp:
    class_indeces = pickle.load(inp)


print(class_indeces.keys())
#**********************

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_generator=test_datagen.flow_from_directory(test_dir,class_mode="categorical", 
                                            target_size=input_shape, 
                                            batch_size=1,classes=class_indeces.keys(),shuffle = False)


###################################################################
loaded_model = tensorflow.keras.models.load_model('Inception_Ourdataset_best_model.epoch22-Loss0.35-Accu0.86.hdf5')

#evaluate
score=loaded_model.evaluate_generator(test_generator)
print("The loss and accuracy:")
print(score)

#Confution Matrix and Classification Report
filenames = test_generator.filenames
num_of_test_samples =len(filenames)
Y_pred = loaded_model.predict(test_generator, num_of_test_samples)
y_pred = np.argmax(Y_pred, axis=1)
y_conf=np.max(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=class_indeces.keys()))

print("The name of files")
#print(filenames)
print("The class of files")
#print(test_generator.classes)
print("The prediction of files")
#print(y_pred)
print("conficence")
#print(y_conf)

