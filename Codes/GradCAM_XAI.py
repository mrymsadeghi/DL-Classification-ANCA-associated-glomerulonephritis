import glob
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#from PIL import Image
import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.models import load_model
from keras.models import load_model
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
import matplotlib.cm as cm


model_path = '/OD-X3_PathD_Imagenet-BG/TwoStep_pathologydataset_Ourdataset_best_model.epoch33-Loss0.18.hdf5'
img_folder_path = '/TestData/patches_bg/'
save_path = "/OD-X3_PathD_Imagenet-BG/XAI_Gradcam_bg" 


def get_img_array(img_path, size): 
    
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    img = cv.resize(img, (size, size))
    img_array = np.array(img)
    print(np.shape(img_array))
    img_array = np.expand_dims(img_array, axis = 0)
    img_array = img_array/ 255.
    return img, img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    return superimposed_img
    # Display Grad CAM
    # display(Image(cam_path))
    # plt.imshow(superimposed_img)


def generate_GradCAM_exp(model, img_path, imgclass, img_size, last_conv_layer_name):
    
    
    img, img_array = get_img_array(img_path, size=img_size)
    model.layers[-1].activation = None
    
    img_class = imgclass
    
    preds = model.predict(img_array)
    prednum = np.argmax(preds, axis=1)
    prediction = 'np'
    if prednum==0:
        prediction = 'crescent'
    elif prednum==1:
        prediction = 'sclerotic'
    elif prednum==2:
        prediction = 'Normal'
        
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    # Display heatmap    
    #plt.matshow(heatmap)    
    superimposed_img =  save_and_display_gradcam(img_path, heatmap)
    
    
    # create figure
    fig, axarr =plt.subplots(1,3, figsize=(30,14))
    fig.patch.set_facecolor('white')

    axarr[0].set_title(f"Label: {str(img_class)} \n prediction: {prediction}\n", fontsize=30)
    axarr[0].imshow(img)
    
    axarr[1].set_title("Heatmap \n", fontsize=30)
    axarr[1].imshow(heatmap)
    
    axarr[2].set_title("GradCAM explanation \n", fontsize=30)
    axarr[2].imshow(superimposed_img)
    
    return fig
    

def batch_gradCAM_exp(model, img_folder_path, save_path, img_size, last_conv_layer_name, labels):

    for label in labels:
        img_files = glob.glob(img_folder_path + label +'/**.png',  recursive=True)        
        for file in img_files:
            print(file)
            img_path = file
            #img_name = img_path.split('/')[-1]
            img_name = img_path.split('/')[-1].split('\\')[-1]
            print(img_name)
            imgclass = img_path.split('/')[-1].split('\\')[0]
            grad_exp = generate_GradCAM_exp(model, img_path, imgclass, img_size, last_conv_layer_name)
            save_folder = os.path.join(save_path, label)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            grad_exp.savefig(os.path.join(save_folder, img_name),transparent=False)
    return

    




last_conv_layer_name = "top_conv"
img_size =240
labels = ['Abnormal - crescent', 'Abnormal - sclerotic', 'Normal']

model = load_model(model_path)

batch_gradCAM_exp(model, img_folder_path, save_path, img_size, last_conv_layer_name, labels)