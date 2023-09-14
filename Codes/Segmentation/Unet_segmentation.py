
import glob
import random
import numpy as np
import cv2 as cv
from keras.metrics import MeanIoU
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.metrics import MeanIoU


imgpath = "Images"
mask_path="Masks"

img_files= sorted(glob.glob(imgpath  + '/**.jpg' , recursive=True))
mask_files =sorted(glob.glob(mask_path  + '/**.jpg' , recursive=True))

imgname = img_files[0].split('/')[-1].split('.')[0]
maskfil = mask_path + "/" + imgname + "_mask.jpg"
imgfil = img_files[0].split('/')[-1].split('.')[0]


def create_data_arr(img_path, mask_path):
    
    SIZE_Y = 256
    SIZE_X = 256
    CHANNEL = 1   
    
    img_files= sorted(glob.glob(img_path  + '/**.jpg' , recursive=True))
    mask_files =sorted(glob.glob(mask_path  + '/**mask.jpg' , recursive=True))

    img_arr=np.zeros((len(img_files),SIZE_Y, SIZE_X, CHANNEL), dtype= np.uint8)
    mask_arr= np.zeros((len(img_files),SIZE_Y, SIZE_X, 1),dtype=np.int16)
    index = 0

    for q in range(len(img_files)):
        
        imgname = img_files[q].split('/')[-1].split('.')[0]
        img_path = img_files[q]
        print(img_path)
        img = cv.imread(img_path, 0)
        img = np.expand_dims(img, axis= -1)
        img_arr[index]=img
        mask_file=mask_path+"/"+ imgname+ "_mask.jpg"
        print(mask_file)
        img_mask = cv.imread(mask_file, 0)
        ret, img_mask= cv.threshold(img_mask,170, 255, cv.THRESH_BINARY)
        img_mask= np.expand_dims(img_mask, axis= -1)
        mask_arr[index] = img_mask

        index=index + 1
    
    return img_arr , mask_arr
    

train_images, train_masks = create_data_arr(imgpath,mask_path )
train_masks = train_masks//255

train_images = np.squeeze(train_images, axis= 3)
train_masks = np.squeeze(train_masks, axis= 3)


labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
print(np.shape(train_masks_reshaped))
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

X_train, X_test, Y_train, Y_test = train_test_split(train_images, train_masks_input, test_size = 0.1, shuffle=True, random_state = 1)

n_classes = 2
train_masks_cat = to_categorical(Y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], n_classes))

test_masks_cat = to_categorical(Y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], n_classes))

#########################################################################
## Training
#########################################################################

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS =X_train.shape[3]


inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
# s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
s = inputs

#Contraction path
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

model_256 = Model(inputs=[inputs], outputs=[outputs])
model_256.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_256.summary()


history = model_256.fit(X_train, y_train_cat, 
                    batch_size = 400, 
                    verbose=1, 
                    epochs=50, 
                    validation_split = 0.2,
                    #validation_data=0.10, 
                    
                    #callbacks=[callback],
                    #class_weight=class_weights,
                    shuffle=True)


model_256.save('U-net Model/model_50_epoches.hdf5')


#########################################################################
## Testing
#########################################################################

pred = model_256.predict(X_test, batch_size = 64, verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')

plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth=Y_test[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model_256.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()



#########################################################################
## IOU performance metric
#########################################################################
y_pred_argmax=np.argmax(pred, axis=3)
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test_cat[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())
#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,1])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[0,1])

print("IoU for class1: Bg is: ", class1_IoU)
print("IoU for class2: Glomeruli is: ", class2_IoU)

plt.imshow(train_images[0, :,:,0], cmap='gray')
plt.imshow(train_masks[0], cmap='gray')