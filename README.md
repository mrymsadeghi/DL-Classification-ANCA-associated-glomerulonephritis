# DL-Classification-ANCA-associated-glomerulonephritis
Dl based models for classifying kidney biopsies of patients with ANCA-associated glomerulonephritis according to the Berden histopathological classification system.





# Summary of the model names

| Dataset w bg | Dataset w/o bg | Pre-trained on |  Architecture  | Img dim |
|:------------:|:--------------:|:--------------:|:--------------:|:-------:|
|      X1_bg   |     X1_nobg    |    Imagenet    |   Inceptionv3  | 150x150 |
|      X2_bg   |     X2_nobg    |    Imagenet    | EfficientNetB1 | 224x224 |
|      X3_bg   |     X3_nobg    |  Imagenet+DPD  | EfficientNetB1 | 224x224 |


DPD: Digital Pathology Dataset (refer to the publication)