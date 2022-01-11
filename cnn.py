import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dropout
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

TRAIN_EPOCHS = 100
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32

#CNN for 3D numpy array
class CNN():
    def __init__(self, input_shape):
        self.model = models.Sequential()

        self.model.add(layers.Conv3D(8, (3, 3, 3) input_shape = input_shape, activation = 'relu', padding='same'))
        self.model.add(layers.Conv3D(16, (3,3,3), activation='relu', padding='same')
        self.model.add(layers.MaxPool3D((2,2,2), padding='same')

        self.model.add(layers.Conv3D(32, (3,3,3), activation='relu', padding='same')
        self.model.add(layers.Conv3D(64, (3,3,3), activation='relu', padding='same')
        self.model.add(layers.MaxPooling3D((2,2,2), padding='same'))

        self.model.add(layers.Conv3D(16, (3,3,3), activation='relu', padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(units=1024, activation='relu'))
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Dense(units=256, activation='relu'))
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Dense(units=10, activation='softmax'))

        #lr=0.001, momentum=0.9
        self.optimizer = optimizers.Adam(lr=0.00001)
        #absolute for regression, squared for classification

        #Absolute for few outliers
        #squared to aggresively diminish outliers
        self.loss = losses.MeanSquaredError()
        #metrics=['accuracy']
        #metrics=['mse']
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def __str__(self):
        self.model.summary(print_fn = self.print_summary)
        return ""

    def print_summary(self, summaryStr):
        print(summaryStr)
