print('[INFO] Importing packages.')
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

from parameters import *        #edit parameters from parameters.py
from cnn import *               #edit cnn from cnn.py

print('[INFO] Done importing packages.')

#checks if GPU is recognized
def checkGPU():
    global devices
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) > 0:
        print('[INFO] GPU is detected.')
    else:
        print('[INFO] GPU not detected.')

#to only save the best model after each epoch
def setCustomCallback():
    global customCallback
    customCallback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        # save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

#won't load all data at once while training and testing
def generator(batchSize, x, y):
    index = 0
    while index < len(x):
        batchX, batchY = [], []
        for i in range(batchSize):
            if (index+i)<len(x):
                batchX.append(x[index+i])
                batchY.append(y[index+i])
                index+=1
            else:
                index=0
        yield np.array(batchX), np.array(batchY)

#load data from folder
def getData(dataPath):
    dataX = []
    dataY = []

    folders = os.listdir(dataPath)
    index = folders.index(".DS_Store")
    folders.pop(index)
    for folder in folders:
        if os.path.isdir(f"{dataPath}/{folder}"):
            for filename in os.listdir(f"{dataPath}/{folder}"):
                filePath = f"{dataPath}/{folder}/{filename}"
                print(filePath)

                with open(filePath, 'rb') as f:
                    numpyArray = np.load(f)
                    print(np.shape(numpyArray))
                    dataX.append(numpyArray)
                    dataY.append(folders.index(folder))

    return dataX, dataY

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def main():
    setCustomCallback()

    dataX, dataY = getData(dataPath)
    # print(dataX, dataY)
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, train_size=0.67, random_state=42)

    cnn = CNN((len(trainX[0]), len(trainX[0][0]), len(trainX[0][0][0]), 1))
    print("[INFO] Printing Tensorflow CNN Summary...")
    print(cnn)

    global results
    results = cnn.model.fit(generator(BATCH_SIZE_TRAIN, trainX, trainY),
        validation_data=generator(BATCH_SIZE_TEST, testX, testY),
        shuffle = True,
        epochs = TRAIN_EPOCHS,
        batch_size = BATCH_SIZE_TRAIN,
        validation_batch_size = BATCH_SIZE_TEST,
        verbose = 1,
        steps_per_epoch=len(trainX)/BATCH_SIZE_TRAIN,
        validation_steps=len(testX)/BATCH_SIZE_TEST,
        callbacks=[customCallback])

if __name__ == "__main__":
    main()
