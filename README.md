# Airbus-Ship-Detection-Challenge

Abstract: this repo includes a pipeline using tf.keras for training UNet for the problem of ships detection.
Moreover, weights and for the trained model are provided. **I will use notebooks/train.ipynb as main file for work.**
Also notebook is better to understand, cause it's sectioned and named. 
*(Also, I explained and commented on most of the stages, since this test is designed more for training - therefore I recommend considering it notebook)*
**+ I had a problem with testing, for some reason the loading of the model does not work correctly for me. I did not have enough time to fix the problem:( Model testing works only in train notebook**

**Important:** balanced dataset (dataset created during analysis) includes 4000 images per each class (0-15 ships) because original dataset contains ~80% images with no ships. Also dataset was downscaled to 256x256, with original resolution the metrics might be better.

### Guide

Important to notice that we have dataset and enviroment so we need to download and install it
   a. We need to create a base directory.
 
   b. Next download dataset from kaggle: [tap here](https://www.kaggle.com/competitions/airbus-ship-detection/data). And unzip it in that folder.
 
   c. Now in base dir must be something like this:
 <pre>
 ├── train_v2
 ├── test_v2
 ├── train_ship_segmentations_v2.csv
 ├── sample_submission_v2.csv
 </pre>
 d. After that we can clone our project files to same folder with data.

     
**Necessary pips** 

```sh
!pip install --user numpy
!pip install pandas
!python -m pip install -U matplotlib
!python -m pip install -U scikit-image
!pip install -U scikit-learn
!pip install keras
!pip install tensorflow
```
Or you can also use requierements.txt.

**Imports:**
```sh
from config import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import binary_opening, disk, label
from PIL import Image
from utils import utils, losses, generators
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy
import keras.backend as K
```

## Architecture:

 - Architecture: UNet
 - Loss function: FocalLoss
 - Optimizer: Adam (lr=1e-3, decay=1e-6)
 - learning scheduler: ReduceLROnPlateau(factor=0.5, patience=3)
 
 ## General thoughts
 
 I've tried DiceBCELoss and DiceLoss, IoU as loss.
 The best results have been obtained with DiceBCELoss in this case.
 
 I need to add I've been bounded with Cuda memory capacity, so basicaly I could not try batch size > 10 with original resolution. So, i said above, it was downscaled.

## Results
| Architecture | binary_accuracy | Input & Mask Resolution | Epochs |
| ------ | ------ | ------ | ------ |
| Unet | ~0.8 - 0.9 | (256x256)  | 6 |
 

