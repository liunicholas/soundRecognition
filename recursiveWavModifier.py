import numpy as np
from scipy.io import wavfile
import wavio
import soundfile

from os import listdir
from os import remove
from os.path import isdir

directory = "soundSamples"
pattern = ".wav"
sampleRate = 11025
lengthSample = 0.5
bitRate = 16

def searchFiles(path, pattern):
    #cycles through all the items in the directory
    for item in listdir(path):
        #if the path is a directory it calls the function again
        if isdir(path + "/" + item):
            searchFiles(path + "/" + item, pattern)
        else:
            if pattern in path + "/" + item:
                makeMonoWav(path + "/" + item)

def makeMonoWav(filePath):
    print(f"making fixed wav file at {filePath}")

    #convert bitRate first
    data, samplerate = soundfile.read(filePath)
    soundfile.write(filePath, data, samplerate, subtype=f'PCM_{bitRate}')

    sample_rate, samples = wavfile.read(filePath)
    print(samples)
    if len(samples) == 0:
        remove(filePath)  #deletes bad file
        return

    print(type(samples[0]))
    #convert stereo to mono
    if str(type(samples[0])) != "<class 'numpy.int16'>":
        x = []
        for pair in samples:
            # avg = (pair[0]+pair[1])/2
            print(pair[0])
            x.append(pair[0])

        x = np.array(x)
        print("converted stereo to mono")
    else:
        x = samples

    notZeroIndex = 0
    while True:
        if x[notZeroIndex] != 0:
            break
        else:
            notZeroIndex += 1

    x = x[notZeroIndex:]
    print(x)

    if len(samples)/sample_rate > lengthSample and sum(abs(x))/len(x) > 10:    #only makes file if the sound is longer than given time
        soundfile.write(filePath, x, sampleRate, subtype=f'PCM_{bitRate}')
    else:
        remove(filePath)

def main():
    searchFiles(directory, pattern)

#must use this for multitprocessing
if __name__ == '__main__':
	main()
