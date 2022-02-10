import matplotlib.pyplot as plt
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
        if max(fftFixed) != 0:
            fftFixed[i]=val/max(fftFixed)
        else:
            fftFixed[i] = 0

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

def convertSpectrogram(audioClip):
    interval = 441           #number of samples to use per fft
    # audioClip = "tools/violin-C4.wav"
    # audioClip = "tools/sine.wav"
    sample_rate, samples = readWavFile(audioClip)

    # print("getting frequencies")
    frequencies = getFreqs(sample_rate)
    # print(f"frequencies:{frequencies}")

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

    return specArray

def main():
    # audioClip = "tools/violin-C4.wav"
    # audioClip = "tools/sine.wav"
    audioClip = "soundSamples/ALARM/Alarm-Fast-High-Pitch-A3-Ring-Tone-www.fesliyanstudios.com.wav"
    convertedWavData = convertSpectrogram(audioClip)

#must use this for multitprocessing
if __name__ == '__main__':
	main()
