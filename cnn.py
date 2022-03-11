import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dropout
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

TRAIN_EPOCHS = 50
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4
soundsClassified = 12

#CNN for 3D numpy array
class CNN():
    def __init__(self, input_shape):
        self.model = models.Sequential()
        # print("here")

        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        # self.model.add(layers.MaxPooling3D((2, 2, 2)))
        # self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # # self.model.add(layers.MaxPooling3D((2, 2, 2)))
        # self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(96, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(layers.Dense(48, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(layers.Dense(24, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(layers.Dense(soundsClassified))

        #lr=0.001, momentum=0.9
        self.optimizer = optimizers.Adam(lr=0.001)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = ['accuracy']

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def __str__(self):
        print(self.model.summary())
        return ""
