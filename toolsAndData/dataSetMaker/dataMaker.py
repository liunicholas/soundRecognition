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
    index = folders.index("PIANO")
    folders.pop(index)
    for folder in folders:
        # print("here")
        if os.path.isdir(f"{directory}/{folder}"):
            # print("here")
            for filename in os.listdir(f"{directory}/{folder}"):
                if filename.endswith(".wav"):
                    print(f"\ngathering data for wav file {counter}")
                    filePath = f"{directory}/{folder}/{filename}"
                    print(f"{filePath}")
                    threeDspec = spectrogramConverter.convertSpectrogram(filePath)
                    threeDspec = np.array(threeDspec)

                    newFileName = Path(filename).stem

                    if not os.path.exists(f"{dataPath}/{folder}"):
                        os.makedirs(f"{dataPath}/{folder}")

                    with open(f"{dataPath}/{folder}/{newFileName}", 'wb') as f:
                        np.save(f, threeDspec)

                    counter+=1

if __name__ == '__main__':
	main()
