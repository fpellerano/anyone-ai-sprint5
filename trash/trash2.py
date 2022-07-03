
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential

import pathlib
dataset_url = "data/car_ims_v1/train"
data_dir = pathlib.Path(dataset_url)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

acura = list(data_dir.glob('Acura_Integra_Type_R_2001/*'))
img = PIL.Image.open(str(acura[1]))
# print(img.show)

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)