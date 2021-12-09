import numpy as np

import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

checkpoint_path = "./untitled folder"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq='epoch')

train = tf.keras.preprocessing.image_dataset_from_directory(
    './soundSamples', labels='inferred', label_mode='categorical',
    class_names=None, color_mode='rgb', batch_size=32, image_size=(128,
    128), shuffle=True, seed=8, validation_split=0.3, subset='training',
    interpolation='bilinear', smart_resize=True
)

validation = tf.keras.preprocessing.image_dataset_from_directory(
    './soundSamples', labels='inferred', label_mode='categorical',
    class_names=None, color_mode='rgb', batch_size=32, image_size=(128,
    128), shuffle=True, seed=8, validation_split=0.7, subset='validation',
    interpolation='bilinear', smart_resize=True
)
