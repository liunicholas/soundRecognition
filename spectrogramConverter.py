import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import wavfile

from math import *
import numpy as np

from multiprocessing import Pool
import scipy.fft as fft

def hardCodeFreqs(baseFreqs):
    # data = read_csv("freqs.csv")
    # freqs = data['Frequency (Hz)'].tolist()

    # baseFreqs = [16.351, 17.324, 18.354, 19.445, 20.601, 21.827, 23.124, 24.499, 25.956, 27.5, 29.135, 30.868]

    binnedFreqs = []
    for freq in baseFreqs:
        currentFreqList = []
        currentFreq = freq
        addFreq = freq
        multiple = 2
        while addFreq < 5012:
            currentFreqList.append(addFreq)
            addFreq = currentFreq*multiple
            multiple+=1
        binnedFreqs.append(currentFreqList)

    return binnedFreqs

def readWavFile(audioClip):
    sample_rate, samples = wavfile.read(audioClip)
    return sample_rate, samples

def getFreqs(sample_rate):
    freqs = fft.fftfreq(sample_rate,1/sample_rate)
    return freqs[:5012]

def getTimesAndSamples(samples, sample_rate, interval):
    times = []
    time = 0.0
    sampleList = []
    for i in range((samples.size-sample_rate)//interval):
        sample = samples[i*interval:i*interval+sample_rate]
        sampleList.append(sample)

        times.append(time)
        time += interval/sample_rate

    return times, sampleList

def getScipyFFT(sample):
    fftResult = fft.fft(sample)
    fftFixed = []
    for val in fftResult[:5012]:
        fftFixed.append(abs(val))

    for i, val in enumerate(fftFixed):
        fftFixed[i]=val/max(fftFixed)

    return fftFixed

def plotSpectrogram(times, frequencies, specArray):
    specArrayT = np.transpose(specArray)

    print(f"times: {times.shape}")
    print(f"frequencies: {frequencies.shape}")
    print(f"spectrogramList: {specArrayT.shape}")
    plt.pcolormesh(times, frequencies, specArrayT)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.savefig("./plots/spectrogram.png")
    plt.show()

def getBinnedSpectrogram(oneInterval, index, x, y, z, c, binnedFreqs, baseFreqs, times):
    threshold = -1
    # x, y, z, c = [], [], [], []
    for i in range(len(oneInterval)):
        for j in range(len(binnedFreqs)):
            for k in range(len(binnedFreqs[j])):
                if i == binnedFreqs[j][k]:
                    if oneInterval[i]>threshold:
                        x.append(times[index])
                        y.append(baseFreqs[j])
                        z.append(k)
                        c.append(oneInterval[i])

    print(f"finished interval {index}")

def multiDimensionPlotting(x, y, z, c):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)

    plt.savefig("./plots/multiDimensionSpectrogram.png")
    plt.show()

def maxLengthListOfList(lists):
    maxLength = 0
    for list in lists:
        if len(list) > maxLength:
            maxLength = len(list)

    return maxLength

def main():
    interval = 441           #number of samples to use per fft
    audioClip = "violin-C4.wav"
    # audioClip = "sine.wav"
    sample_rate, samples = readWavFile(audioClip)

    print("getting frequencies")
    frequencies = getFreqs(sample_rate)
    print(f"frequencies:{frequencies}")

    #gets groups of samples and the times that each sample starts at
    times, sampleList = getTimesAndSamples(samples, sample_rate, interval)
    with Pool(processes=8, maxtasksperchild = 1) as pool:
            print("making fft for samples")
            spectrogramList = pool.map(getScipyFFT, sampleList)
            pool.close()
            pool.join()

    times = np.array(times)
    frequencies = np.array(frequencies)
    specArray = np.array(spectrogramList)

    # plots spectrogram
    # plotSpectrogram(times, frequencies, specArray)

    print("making base and binned frequencies")
    baseFreqs = range(100,200)
    #bin frequncies using base frequencies
    binnedFreqs = hardCodeFreqs(baseFreqs)

    # plot single frame
    # x, y, z, c = [], [], [], []
    # getBinnedSpectrogram(specArray[0], 0, x, y, z, c, binnedFreqs, baseFreqs, times)

    # multiDimensionPlotting(x,y,z,c)

    print("organizing lists for machine learning")
    #find most amount of harmonics in a list
    lengthList = maxLengthListOfList(binnedFreqs)
    print(f"length of each row: {lengthList}")

    # print(spectrogramOneInterval)
    # length = len(spectrogramOneInterval[0])
    # for list in spectrogramOneInterval:
    #     assert len(list) == length

    # get every frame
    convertedWavData = []
    for index in range(len(specArray)):
        x, y, z, c = [], [], [], []
        getBinnedSpectrogram(specArray[index], index, x, y, z, c, binnedFreqs, baseFreqs, times)

        # get a list of lists and each list represents a base frequency
        spectrogramOneInterval = []
        for i in range(len(baseFreqs)):
            validPointsAtBaseFreq = []
            for o in range(len(y)):
                if y[o] == baseFreqs[i]:
                    validPointsAtBaseFreq.append(c[o])

            baseFreqInterval = validPointsAtBaseFreq
            for j in range(lengthList-len(validPointsAtBaseFreq)):
                baseFreqInterval.append(0)

            spectrogramOneInterval.append(baseFreqInterval)

        convertedWavData.append(spectrogramOneInterval)

    convertedWavData = np.array(convertedWavData)
    print(f"shape of data: {convertedWavData.shape}")

#must use this for multitprocessing
if __name__ == '__main__':
	main()
