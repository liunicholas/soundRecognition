import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import h5py

#source test data: https://www.kaggle.com/daavoo/3d-mnist

 with h5py.File("../input/train_point_clouds.h5", "r") as hf:
     X_train = hf["X_train"][:]
     y_train = hf["y_train"][:]
     X_test = hf["X_test"][:]
     y_test = hf["y_test"][:]

"""
sample code for 3d?
model = Sequential([
    layers.Conv3D(8, (3,3,3), activation='relu', padding='same', input_shape=(1, 16,16,16)),
    layers.Conv3D(16, (3,3,3), activation='relu', padding='same'),
    layers.MaxPooling3D((2,2,2), padding='same'),

    layers.Conv3D(32, (3,3,3), activation='relu', padding='same'),
    layers.Conv3D(64, (3,3,3), activation='relu', padding='same'),
    layers.MaxPooling3D((2,2,2), padding='same'),

    layers.Conv3D(16, (3,3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling3D(),
    layers.Flatten(),

    layers.Dense(units=1024, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(units=256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(units=10, activation='softmax'),
])
"""
