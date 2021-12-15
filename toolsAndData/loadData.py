import numpy as np
import os

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
    
def main():
    dataPath = "spectrogramData"
    dataX, dataY = getData(dataPath)
    print(dataX)
    print(dataY)

main()
