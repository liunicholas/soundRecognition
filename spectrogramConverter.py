import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# import librosa
# import librosa.display

from math import *
import numpy as np
from pandas import *

import scipy.fft as fft

def librosaStuff(audioClip):
    x, sr = librosa.load(audioClip, sr=11025)
    x = x[:sr]
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    # plt.show()

    thing = librosa.feature.melspectrogram(x, sr)
    print(thing)
    Nfft = 256
    stft = librosa.stft(x, n_fft=Nfft, window=sig.windows.hamming)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=Nfft)
    #
    # freqs = librosa.fft_frequencies(sr=11025, n_fft=2048)

    # freqs = librosa.mel_frequencies(n_mels=5512)
    print(len(freqs))

    X = librosa.stft(x)
    Xdb = (librosa.amplitude_to_db(abs(X)))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    # plt.show()
def multiDimensionPlotting():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.random.standard_normal(100)
    y = np.random.standard_normal(100)
    z = np.random.standard_normal(100)
    c = np.random.standard_normal(100)

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()
def sciPySpectrogram(audioClip):
    sample_rate, samples = wavfile.read('/Users/nicholasliu/Documents/adhoncs/soundRecognition/violin-C4.wav')
    print(len(samples))
    sampleList = []
    for i in range(samples.size//sample_rate):
        sampleList.append(samples[:sample_rate])
        samples = samples[sample_rate+1:]
    # print(len(sampleList[0]))
    frequencies, times, spectrogram = signal.spectrogram(sampleList[0], fs = 1/len(sampleList[0]))

    print("frequencies:")
    print(frequencies)
    print("time:")
    print(len(times))
    print("spectrogram:")
    print(len(spectrogram))

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

#https://stackoverflow.com/a/64505498
def frequency_to_note(frequency):
    # define constants that control the algorithm
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] # these are the 12 notes in each octave
    OCTAVE_MULTIPLIER = 2 # going up an octave multiplies by 2
    KNOWN_NOTE_NAME, KNOWN_NOTE_OCTAVE, KNOWN_NOTE_FREQUENCY = ('A', 4, 440) # A4 = 440 Hz

    # calculate the distance to the known note
    # since notes are spread evenly, going up a note will multiply by a constant
    # so we can use log to know how many times a frequency was multiplied to get from the known note to our note
    # this will give a positive integer value for notes higher than the known note, and a negative value for notes lower than it (and zero for the same note)
    note_multiplier = OCTAVE_MULTIPLIER**(1/len(NOTES))
    frequency_relative_to_known_note = frequency / KNOWN_NOTE_FREQUENCY
    distance_from_known_note = math.log(frequency_relative_to_known_note, note_multiplier)

    # round to make up for floating point inaccuracies
    distance_from_known_note = round(distance_from_known_note)

    # using the distance in notes and the octave and name of the known note,
    # we can calculate the octave and name of our note
    # NOTE: the "absolute index" doesn't have any actual meaning, since it doesn't care what its zero point is. it is just useful for calculation
    known_note_index_in_octave = NOTES.index(KNOWN_NOTE_NAME)
    known_note_absolute_index = KNOWN_NOTE_OCTAVE * len(NOTES) + known_note_index_in_octave
    note_absolute_index = known_note_absolute_index + distance_from_known_note
    note_octave, note_index_in_octave = note_absolute_index // len(NOTES), note_absolute_index % len(NOTES)
    note_name = NOTES[note_index_in_octave]
    return (note_name, note_octave)
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
def binFreqs(freqs):
    #lowest frequency of a series of multiples, this is the x value
    baseFreqs = []
    #index of the frequencies of harmonic series,
    harmonicIndexes = []
    #dictionary for each frequency to which baseFreq
    harmonicDict = {}
    for i in range(len(freqs)):
        ISBASE = False
        for j in range(len(baseFreqs)):
            if baseFreqs[j]*1 != 0:
                # if len(baseFreqs)<=12:
                #     errorThreshold = 0.5
                # else:
                #     errorThreshold = 2
                errorThreshold = 0.5
                if freqs[i] % baseFreqs[j] <= errorThreshold:
                    print(baseFreqs[j])
                    print(freqs[i])
                    print(freqs[i] % baseFreqs[j])
                    harmonicIndexes[j].append(i)
                    harmonicDict.update({freqs[i]:j})
                    ISBASE = True
                    break
        if ISBASE:
            continue

        print("Found new base frequency.")
        baseFreqs.append(freqs[i])
        harmonicIndexes.append([i])
        harmonicDict.update({freqs[i]:i})

    return baseFreqs, harmonicIndexes, harmonicDict

def readWavFile(audioClip):
    sample_rate, samples = wavfile.read(audioClip)
    return sample_rate, samples

def getFreqs(sample_rate):
    freqs = fft.fftfreq(sample_rate,1/sample_rate)
    return freqs[:5012]

def getScipyFFT(sample):
    fftResult = fft.fft(sample)
    fftFixed = []
    for val in fftResult[:5012]:
        fftFixed.append(abs(val))
    # print(max(fftFixed))
    # print(min(fftFixed))
    for i, val in enumerate(fftFixed):
        fftFixed[i]=val/max(fftFixed)
    # for i, val in enumerate(fftFixed):
    #     if val > 0.95:
    #         print(i)

    return fftFixed

def main():
    interval = 441
    audioClip = "violin-C4.wav"
    sample_rate, samples = readWavFile(audioClip)

    frequencies = getFreqs(sample_rate)
    print(f"frequencies:{frequencies}")
    baseFreqs, harmonicIndexes, harmonicDict = binFreqs(frequencies)
    print(baseFreqs)

    times = []
    time = 0.0
    spectrogramList = []
    for i in range((samples.size-sample_rate)//interval):
        sample = samples[i*interval:i*interval+sample_rate]
        fftFixed = getScipyFFT(sample)
        spectrogramList.append(fftFixed)

        times.append(time)
        time += interval/sample_rate
        print(i)

    print(f"times: {len(times)}")
    print(f"frequencies: {len(frequencie}")
    print(f"spectrogramListOneList: {len(spectrogramList[0])}")
    plt.pcolor(times, frequencies, spectrogramList)
    plt.imshow(spectrogramList)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    # plt.show()
    plt.savefig("./plots/spectrogram.png")





main()
