import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.io import wavfile

from math import *
import numpy as np

from multiprocessing import Pool
import scipy.fft as fft

fftCount = 0

def hardCodeFreqs():
    data = read_csv("freqs.csv")
    freqs = data['Frequency (Hz)'].tolist()

    baseFreqs = [16.351, 17.324, 18.354, 19.445, 20.601, 21.827, 23.124, 24.499, 25.956, 27.5, 29.135, 30.868]

    allFreqs = []
    for freq in baseFreqs:
        currentFreqList = []
        currentFreq = freq
        addFreq = freq
        multiple = 2
        while addFreq < 11025:
            currentFreqList.append(addFreq)
            addFreq = currentFreq*multiple
            multiple+=1
        allFreqs.append(currentFreqList)

    print(allFreqs[0])

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
    print(f"times: {times.shape}")
    print(f"frequencies: {frequencies.shape}")
    print(f"spectrogramList: {specArray.shape}")
    plt.pcolormesh(times, frequencies, specArray)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    plt.savefig("./plots/spectrogram.png")

def multiDimensionPlotting(x, y, z, c):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
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
    specArray = np.transpose(specArray)

    plotSpectrogram(times, frequencies, specArray)

#must use this for multitprocessing
if __name__ == '__main__':
	main()
