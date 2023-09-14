
# DL-Classification-ANCA-associated-glomerulonephritis
Dl based models for classifying kidney biopsies of patients with ANCA-associated glomerulonephritis according to the Berden histopathological classification system.


## Structure of files and folders
 In the codes folder you will find codes for training and testing the classification models X1, X2 and X3 (find specifications below) and the segmentation UNet model. Also the code which was used to extract the GradCam data, is provided.

 In the Images folder you will find sample patches of training data, with and without back ground. Also there are sample images of GradCAM algorithm output as well as the performance of the segmentation algorithm


## Summary of the model names

| Dataset w bg | Dataset w/o bg | Pre-trained on |  Architecture  | Img dim |
|:------------:|:--------------:|:--------------:|:--------------:|:-------:|
|      X1_bg   |     X1_nobg    |    Imagenet    |   Inceptionv3  | 150x150 |
|      X2_bg   |     X2_nobg    |    Imagenet    | EfficientNetB1 | 224x224 |
|      X3_bg   |     X3_nobg    |  Imagenet+DPD  | EfficientNetB1 | 224x224 |


DPD: Digital Pathology Dataset (refer to the publication)


## Publication

Will be updated soon.

## TRained Models

The trained models can be found in the google drive link below.
https://drive.google.com/file/d/1DZ9fQteJYvHt9gjoWxQOrn1GolDCJGdw/view?usp=sharing
