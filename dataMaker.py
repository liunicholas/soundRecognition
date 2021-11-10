import numpy as np
import os
import spectrogramConverter

directory = "mono samples"
dataPath = "allSpectrogramData"
allData = []
counter = 1

def main():
    global counter
    wavFiles = os.listdir(directory)
    for WAV_FILENAME in wavFiles:
        print(f"gathering data for wav file {counter}")
        filePath = os.path.join(directory, WAV_FILENAME)

        fourDspec = spectrogramConverter.convertSpectrogram(filePath)
        allData.append(fourDspec)

        counter+=1

    allData = np.array(allData)
    with open(dataPath, 'wb') as f:
        np.save(f, allData)

if __name__ == '__main__':
	main()
