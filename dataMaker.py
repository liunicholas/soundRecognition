import numpy as np
import os
import spectrogramConverter
from pathlib import Path

directory = "soundSamples"
dataPath = "spectrogramData"
allData = []
counter = 1

def main():
    global counter
    folders = os.listdir(directory)
    for folder in folders:
        for wavFile in folder:
        print(f"\ngathering data for wav file {counter}")
        filePath = f"{directory}/{folder}/{wavFile}"
        fourDspec = spectrogramConverter.convertSpectrogram(filePath)
        fourDspec = np.array(fourDspec)

        newFileName = Path(wavFile).stem
        with open(f"{dataPath}/{folder}/{newFileName}", 'wb') as f:
            np.save(f, fourDspec)

        counter+=1

if __name__ == '__main__':
	main()
