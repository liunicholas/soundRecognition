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
        # For Conv2D, you give it: Outgoing Layers, Frame size.  Everything else needs a keyword.
        # Popular keyword choices: strides (default is strides=1), padding (="valid" means 0, ="same" means whatever gives same output width/height as input).  Not sure yet what to do if you want some other padding.
        # Activation function is built right into the Conv2D function as a keyword argument.

        self.model.add(layers.Conv1D(32, 3, input_shape = input_shape, activation = 'relu'))
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # self.model.add(layers.BatchNormalization(trainable=False))
        # self.model.add(Dropout(0.05))

        # self.model.add(layers.MaxPooling2D(pool_size = 2))

        self.model.add(layers.Conv1D(16, 3, activation = 'relu'))
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # self.model.add(layers.BatchNormalization(trainable=False))
        # self.model.add(Dropout(0.1))

        # self.model.add(layers.Conv1D(128, 3, activation = 'relu'))
        # self.model.add(layers.BatchNormalization(trainable=False))
        # self.model.add(Dropout(0.15))

        # self.model.add(layers.Conv1D(256, 3, activation = 'relu'))
        # self.model.add(layers.BatchNormalization(trainable=False))
        # self.model.add(Dropout(0.2))

        # self.model.add(layers.MaxPooling2D(pool_size = 2))

        self.model.add(layers.Flatten())

        #get to one value
        # self.model.add(layers.Dense(2400, activation = 'relu', input_shape = input_shape))
        # self.model.add(layers.Dense(1200, activation = 'relu'))
        # self.model.add(layers.Dense(600, activation = 'relu'))
        # self.model.add(layers.Dense(300, activation = 'relu'))
        # self.model.add(layers.Dense(120, activation = 'relu'))
        # self.model.add(layers.Dense(60, activation = 'relu'))
        # self.model.add(layers.Dense(20, activation = 'relu'))
        # self.model.add(layers.Dense(1))

        # self.model.add(layers.Dense(32, activation = 'relu', input_shape = input_shape))
        # self.model.add(layers.Dense(16, activation = 'relu'))
        self.model.add(layers.Dense(1))

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

# imports
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from tensorflow.keras.utils import to_categorical
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Model configuration
batch_size = 100
no_epochs = 30
learning_rate = 0.001
no_classes = 10
validation_split = 0.2
verbosity = 1




# Convert 1D vector into 3D values, provided by the 3D MNIST authors at
# https://www.kaggle.com/daavoo/3d-mnist
# def array_to_color(array, cmap="Oranges"):
#   s_m = plt.cm.ScalarMappable(cmap=cmap)
#   return s_m.to_rgba(array)[:,:-1]

# Reshape data into format that can be handled by Conv3D layers.
# Courtesy of Sam Berglin; Zheming Lian; Jiahui Jang - University of Wisconsin-Madison
# Report - https://github.com/sberglin/Projects-and-Papers/blob/master/3D%20CNN/Report.pdf
# Code - https://github.com/sberglin/Projects-and-Papers/blob/master/3D%20CNN/network_final_version.ipynb
# def rgb_data_transform(data):
#   data_t = []
#   for i in range(data.shape[0]):
#     data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
#   return np.asarray(data_t, dtype=np.float32)



# -- Process code --
# Load the HDF5 data file
# with h5py.File("./full_dataset_vectors.h5", "r") as hf:

    # Split the data into training/test features/targets
    # X_train = hf["X_train"][:]
    # targets_train = hf["y_train"][:]
    # X_test = hf["X_test"][:]
    # targets_test = hf["y_test"][:]

    # Determine sample shape
    # sample_shape = (62, 50, 100, 3)

    # Reshape data into 3D format
    # X_train = rgb_data_transform(X_train)
    # X_test = rgb_data_transform(X_test)

    # Convert target vectors to categorical targets
    # targets_train = to_categorical(targets_train).astype(np.integer)
    # targets_test = to_categorical(targets_test).astype(np.integer)

# Create the model
# shape of data: 62, 50, 100

tf.keras.layers.Conv3D(
    filters, kernel_size=3, strides=1, padding='valid',
    data_format=None, dilation_rate=1, groups=1, activation='relu',
    use_bias=True, kernel_initializer='he_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)


model = Sequential()
model.add(Conv3D(32, input_shape(10, 100, 50, 1)))
#8, 98, 48, 32
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#30, 29, 49, 32
model.add(Conv3D(64))
#28, 27, 47, 64
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#14, 14, 24, 64
model.add(Conv3D(128))
#12, 12, 22, 128
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(no_classes, activation='softmax'))


# Compile the model
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

# Fit data to model
history = model.fit(X_train, targets_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)

# Generate generalization metrics
# score = model.evaluate(X_test, targets_test, verbose=0)
# print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Plot history: Categorical crossentropy & Accuracy
# plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
# plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
# plt.plot(history.history['accuracy'], label='Accuracy (training data)')
# plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
# plt.title('Model performance for 3D MNIST Keras Conv3D example')
# plt.ylabel('Loss value')
# plt.xlabel('No. epoch')
# plt.legend(loc="upper left")
# plt.show()
