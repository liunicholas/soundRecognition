import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dropout
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

TRAIN_EPOCHS = 10
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4
soundsClassified = 12

#CNN for 2D numpy array
class CNN():
    def __init__(self, input_shape):
        self.model = models.Sequential([
        self.model.add(layers.Input(shape=input_shape))
        self.layers.Resizing(32, 32)
        self.norm_layer
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels)



        self.model = models.Sequential()

        self.model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape))
        # self.model.add(layers.MaxPooling3D((2, 2, 2)))
        # self.model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
        # self.model.add(layers.MaxPooling3D((2, 2, 2)))
        # self.model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(soundsClassified))

        #lr=0.001, momentum=0.9
        self.optimizer = optimizers.Adam(lr=0.00001)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = ['accuracy']
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def __str__(self):
        self.model.summary(print_fn = self.print_summary)
        return ""

    def print_summary(self, summaryStr):
        print(summaryStr)
