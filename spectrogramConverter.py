import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
from scipy import signal

from math import *
import numpy as np

from multiprocessing import Pool
import scipy.fft as fft

# fftCount = 0

def sciPySpectrogram(audioClip):
    sample_rate, samples = wavfile.read(audioClip)
    print(len(samples))

    plt.plot(np.arange(1,len(samples)+1), samples)
    plt.show()

    sampleList = []
    for i in range(samples.size//sample_rate):
        sampleList.append(samples[:sample_rate])
        samples = samples[sample_rate+1:]
    # print(len(sampleList[0]))
    frequencies, times, spectrogram = signal.spectrogram(sampleList[1], fs = 1/len(sampleList[0]))

    print("frequencies:")
    print(frequencies.shape)
    print("time:")
    print(times.shape)
    print("spectrogram:")
    print(spectrogram.shape)

    plt.pcolormesh(times, frequencies, spectrogram, shading = 'auto')
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

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
    # global fftCount
    fftResult = fft.fft(sample)
    fftFixed = []
    for val in fftResult[:5012]:
        fftFixed.append(abs(val))

    for i, val in enumerate(fftFixed):
        fftFixed[i]=val/max(fftFixed)

    # print(fftCount)
    # fftCount+=1

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
    # x, y, z, c = [], [], [], []
    for i in range(len(oneInterval)):
        for j in range(len(binnedFreqs)):
            for k in range(len(binnedFreqs[j])):
                if i == binnedFreqs[j][k]:
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

def getOneSpectrogram(testParams):
    oneInterval, index, binnedFreqs, baseFreqs, times = testParams[0], testParams[1], testParams[2], testParams[3], testParams[4]
    x, y, z, c = [], [], [], []
    for i in range(len(oneInterval)):
        for j in range(len(binnedFreqs)):
            for k in range(len(binnedFreqs[j])):
                if i == binnedFreqs[j][k]:
                    x.append(times[index])
                    y.append(baseFreqs[j])
                    z.append(k)
                    c.append(oneInterval[i])

    return([x,y,z,c])

def init():
    return threeDeeSpectrogram[0]

def animate(i):
    return threeDeeSpectrogram[i]

def animationFunction(threeDeeSpectrogram):
    fig = plt.figure()
    ax = Axes3D(fig)
    anim = FuncAnimation(fig, animate, init_func=init,
        frames=len(threeDeeSpectrogram), interval=20, blit=False)
    plt.show()

def main():
    interval = 441           #number of samples to use per fft
    audioClip = "violin-C4.wav"
    # audioClip = "sine.wav"
    sample_rate, samples = readWavFile(audioClip)

    # basic spectrogram using scipy signal
    # sciPySpectrogram(audioClip)

    print("getting frequencies")
    frequencies = getFreqs(sample_rate)
    print(f"frequencies:{frequencies}")

    #gets groups of samples and the times that each sample starts at
    times, sampleList = getTimesAndSamples(samples, sample_rate, interval)
    print("made times and sample list")
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

    # x, y, z, c = [], [], [], []
    # plot every frame at once
    # for index in range(len(specArray)):
    #     getBinnedSpectrogram(specArray[index], index, x, y, z, c, binnedFreqs, baseFreqs, times)

    print("creating 3d spectrogram at each interval")
    spectrogramGroupings = []
    for index in range(len(specArray)):
        oneGroup = []
        oneGroup.append(specArray[index])
        oneGroup.append(index)
        oneGroup.append(binnedFreqs)
        oneGroup.append(baseFreqs)
        oneGroup.append(times)
        spectrogramGroupings.append(oneGroup)

    # print(spectrogramGroupings)

    # global threeDeeSpectrogram
    # with Pool(processes=8, maxtasksperchild = 1) as pool:
    #         threeDeeSpectrogram = pool.map(getOneSpectrogram, spectrogramGroupings)
    #         pool.close()
    #         pool.join()
    global threeDeeSpectrogram
    threeDeeSpectrogram = []
    for i in range(len(spectrogramGroupings)):
        threeDeeSpectrogram.append(getOneSpectrogram(spectrogramGroupings[i]))

    print(threeDeeSpectrogram)
    animationFunction(threeDeeSpectrogram)

    # threeDeeSpectrogram = []
    # for index in range(len(specArray)):
    #     threeDeeSpectrogram.append(getOneSpectrogram(specArray[index], index, binnedFreqs, baseFreqs, times))

    # plot single frame
    # getBinnedSpectrogram(specArray[0], 0, x, y, z, c, binnedFreqs, baseFreqs, times)

    # x, y, z, c = listResults[0], listResults[1], listResults[2], listResults[3]
    # multiDimensionPlotting(x,y,z,c)

#must use this for multitprocessing
if __name__ == '__main__':
	main()
