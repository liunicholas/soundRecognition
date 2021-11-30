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
        for filename in os.listdir(f"{directory}/{folder}"):
            if filename.endswith(".wav"):
                print(f"\ngathering data for wav file {counter}")
                filePath = f"{directory}/{folder}/{filename}"
                print(f"{filePath}")
                fourDspec = spectrogramConverter.convertSpectrogram(filePath)
                fourDspec = np.array(fourDspec)

                newFileName = Path(filename).stem
                
                if not os.path.exists(f"{dataPath}/{folder}"):
                    os.makedirs(f"{dataPath}/{folder}")

                with open(f"{dataPath}/{folder}/{newFileName}", 'wb') as f:
                    np.save(f, fourDspec)

                counter+=1

if __name__ == '__main__':
	main()
