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
