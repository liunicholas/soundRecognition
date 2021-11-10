import numpy as np
import os
from scipy.io import wavfile
import wavio
import time
from multiprocessing import Pool

directory = "samples"
newDirectory = "mono samples"

def makeMonoWav(WAV_FILENAME):
    filePath = os.path.join(directory, WAV_FILENAME)

    sample_rate, samples = wavfile.read(filePath)

    x = []
    for pair in samples:
        # avg = (pair[0]+pair[1])/2
        print(pair[0])
        x.append(pair[0])

    x = np.array(x)

    wavio.write(os.path.join(newDirectory, f"{WAV_FILENAME}"), x, 11025, sampwidth=3)

def main():
    wavFiles = os.listdir(directory)
    with Pool(processes=8, maxtasksperchild = 1) as pool:
            results = pool.map(makeMonoWav, wavFiles)
            pool.close()
            pool.join()

#must use this for multitprocessing
if __name__ == '__main__':
	main()
