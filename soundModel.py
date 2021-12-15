print('[INFO] Importing packages.')
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
from datetime import *
from os import mkdir

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

def getData(dataPath):
    dataX = []
    dataY = []

    folders = os.listdir(dataPath)
    for folder in folders:
        if os.path.isdir(f"{dataPath}/{folder}"):
            for filename in os.listdir(f"{dataPath}/{folder}"):
                filePath = f"{dataPath}/{folder}/{filename}"
                print(filePath)

                with open(filePath, 'rb') as f:
                    numpyArray = np.load(f)
                    print(np.shape(numpyArray))
                    dataX.append(numpyArray)
                    dataY.append(folder)

    return dataX, dataY

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

#ask user if they want to save the model
def askUserSaveModel():
    if remoteMachine:
        return "y"

    else:
        while True:
            keep = input("save this model to folder? (y/n)")
            if keep != "y" and keep != "n":
                print("error")
                continue
            break

        return keep
#ask user for version name
def getVersionName():
    if remoteMachine:
        return remoteVersionName

    else:
        while True:
            version = input("version name: ")
            while True:
                confirm = input("confirm? (y/n)")
                if confirm != "y" and confirm != "n":
                    print("error")
                    continue
                break
            if confirm == "y":
                break

        return version

#makes new folder for saved model
def makeNewFolder(version):
    print("[INFO] Making New Model Folder.")
    newFolderPath = f"{savedModelsPath}/{daysBefore}_{daysAhead}_{version}"
    mkdir(newFolderPath)

    return newFolderPath
#saves pyplot to folder for later analysis
def savePyPlot(newFolderPath, version, holdoutItems, testItems, trainItems):
    print("[INFO] Saving Pyplot.")
    fig = getLossAndPriceGraph(results, holdoutItems, testItems, trainItems)
    plt.savefig(f"{newFolderPath}/{daysBefore}_{daysAhead}_{version}.png")
#saves text file of included stocks to folder
def saveIncludedStocks(newFolderPath):
    print("[INFO] Saving Included Stocks Text File.")
    stocksIncluded = readFile(stocksIncludedPath)
    writeFile(f"{newFolderPath}/stocksIncluded.txt", stocksIncluded)
#saves best model to folder
def saveModel(newFolderPath, bestModel):
    print("[INFO] Saving Model.")
    bestModel.save(newFolderPath)
#saves all info to text file
def saveParameters(newFolderPath, version, numStocks):
    print("[INFO] Saving Parameters.")
    f = open(f"{newFolderPath}/{daysBefore}_{daysAhead}_{version}_info.txt", 'w')
    f.write(f"version name: {daysBefore}_{daysAhead}_{version}\n")
    f.write(f"training dates: {trainStart} to {trainEnd}\n")
    f.write(f"testing dates: {testStart} to {testEnd}\n")
    f.write(f"holdout dates: {holdoutStart} to {holdoutEnd}\n")
    f.write(f"days before: {daysBefore}\n")
    f.write(f"days ahead: {daysAhead}\n")
    f.write(f"number of stocks included: {numStocks}\n")
    f.close()

#get loss and price graph
def getLossAndPriceGraph(results, holdoutItems, testItems, trainItems):
    #index 0 is dates, index 1 is real, index 2 is predictions
    fig = plt.figure("preds vs real high price", figsize=(15, 8))
    fig.tight_layout()
    #training and validation loss
    plt1 = fig.add_subplot(221)
    plt1.title.set_text("training and validation loss")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="training")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="validation")
    plt1.legend(loc='upper right')
    #training set
    plt2 = fig.add_subplot(222)
    plt2.title.set_text("training set real and preds")
    plt2.plot(trainItems[0], trainItems[1], color="blue", label="real")
    plt2.plot(trainItems[0], trainItems[2], color="red", label="preds")
    plt2.legend(loc='upper left')
    #validation set
    plt3 = fig.add_subplot(223)
    plt3.title.set_text("validation set real and preds")
    plt3.plot(testItems[0], testItems[1], color="blue", label="real")
    plt3.plot(testItems[0], testItems[2], color="red", label="preds")
    plt3.legend(loc='upper left')
    #holdout set
    plt4 = fig.add_subplot(224)
    plt4.title.set_text("holdout set real and preds")
    plt4.plot(holdoutItems[0], holdoutItems[1], color="blue", label="real")
    plt4.plot(holdoutItems[0], holdoutItems[2], color="red", label="preds")
    plt4.legend(loc='upper left')

    return fig
#get real vs preds price graph
def getJustPriceGraph(holdoutItems, testItems):
    fig = plt.figure("preds vs real high price", figsize=(10, 8))
    fig.tight_layout()
    #validation set
    plt1 = fig.add_subplot(211)
    plt1.title.set_text("validation set real and preds")
    plt1.plot(testItems[0], testItems[1], color="blue", label="real")
    plt1.plot(testItems[0], testItems[2], color="red", label="preds")
    plt1.legend(loc='upper left')
    #holdout set
    plt2 = fig.add_subplot(212)
    plt2.title.set_text("holdout set real and preds")
    plt2.plot(holdoutItems[0], holdoutItems[1], color="blue", label="real")
    plt2.plot(holdoutItems[0], holdoutItems[2], color="red", label="preds")
    plt2.legend(loc='upper left')

    return fig


#train model with CNN
def train():
    dataPath = "spectrogramData"
    dataX, dataY = getData(dataPath)


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

#make predictions on old data
def test():

    bestModel = tf.keras.models.load_model(checkpointPath)

    print(f"[INFO] Making Predictions.")
    holdoutPredictions = bestModel.predict(holdoutX)
    testPredictions = bestModel.predict(testX)
    trainPredictions = bestModel.predict(trainX)

    # displayPredictionsAsText(histHoldoutIndex, predictions)

    holdoutItems = [histHoldoutIndex, holdoutY, holdoutPredictions]
    testItems = [histTestIndex, testY, testPredictions]
    trainItems = [histTrainIndex, trainY, trainPredictions]

    if TRAIN:
        fig = getLossAndPriceGraph(results, holdoutItems, testItems, trainItems)
        plt.savefig(graphPath)
    else:
        fig = getJustPriceGraph(holdoutItems, testItems)
        plt.savefig(graphPath)

    if not remoteMachine:
        plt.show()
        print(f"[INFO] Close plot to continue.")

    #ask to save model if new model
    if NEW_MODEL:
        keep = askUserSaveModel()
        if keep == "y":
            version = getVersionName()

            newFolderPath = makeNewFolder(version)
            savePyPlot(newFolderPath, version, holdoutItems, testItems, trainItems)
            saveIncludedStocks(newFolderPath)
            saveModel(newFolderPath, bestModel)

            numStocks = getNumStocks(testX, trainX, holdoutX)
            saveParameters(newFolderPath, version, numStocks)

#predict sound
def predictSound():
    return

def main():
    checkGPU()
    setModes()
    setCustomCallback()

    if LOAD_DATASET:
        loadData()
    if TRAIN:
        train()
    if TEST:
        test()
    if PREDICT_ON_DATE:
        PredictOnDate()

if __name__ == "__main__":
    main()
