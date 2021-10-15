import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import wavfile

from math import *
import numpy as np

from multiprocessing import Pool
import scipy.fft as fft

fftCount = 0

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

    # print(binnedFreqs[0])

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
    global fftCount
    fftResult = fft.fft(sample)
    fftFixed = []
    for val in fftResult[:5012]:
        fftFixed.append(abs(val))

    for i, val in enumerate(fftFixed):
        fftFixed[i]=val/max(fftFixed)

    print(fftCount)
    fftCount+=1

    return fftFixed

def plotSpectrogram(times, frequencies, specArray):
    specArray = np.transpose(specArray)

    print(f"times: {times.shape}")
    print(f"frequencies: {frequencies.shape}")
    print(f"spectrogramList: {specArray.shape}")
    plt.pcolormesh(times, frequencies, specArray)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.savefig("./plots/spectrogram.png")
    plt.show()

def multiDimensionPlotting(x, y, z, c):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)

    plt.savefig("./plots/multiDimensionSpectrogram.png")
    plt.show()

def main():
    interval = 441           #number of samples to use per fft
    audioClip = "violin-C4.wav"
    sample_rate, samples = readWavFile(audioClip)

    frequencies = getFreqs(sample_rate)
    print(f"frequencies:{frequencies}")

    times, sampleList = getTimesAndSamples(samples, sample_rate, interval)
    with Pool(processes=8, maxtasksperchild = 1) as pool:
            spectrogramList = pool.map(getScipyFFT, sampleList)
            pool.close()
            pool.join()

    times = np.array(times)
    frequencies = np.array(frequencies)
    specArray = np.array(spectrogramList)

    # plotSpectrogram(times, frequencies, specArray)

    baseFreqs = range(100,200)
    binnedFreqs = hardCodeFreqs(baseFreqs)

    x, y, z, c = [], [], [], []
    onePart = specArray[0]
    for i in range(len(onePart)):
        for j in range(len(binnedFreqs)):
            for k in range(len(binnedFreqs[j])):
                if i == binnedFreqs[j][k]:
                    x.append(0)
                    y.append(baseFreqs[j])
                    z.append(k)
                    c.append(specArray[0][i])

    multiDimensionPlotting(x,y,z,c)

#must use this for multitprocessing
if __name__ == '__main__':
	main()
