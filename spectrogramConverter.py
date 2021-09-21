import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

import librosa
import librosa.display

audioClip = "/Users/nicholasliu/Documents/adhoncs/soundRecognition/violin-C4.wav"

x, sr = librosa.load(audioClip, sr=11025)
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr)
# plt.show()

# thing = librosa.feature.melspectrogram(x, sr)
# print(thing)
# Nfft = 256
# stft = librosa.stft(x, n_fft=Nfft, window=sig.windows.hamming)
# freqs = librosa.fft_frequencies(sr=sr, n_fft=Nfft)

freqs = librosa.fft_frequencies(sr=11025, n_fft=2048)

# freqs = librosa.mel_frequencies(n_mels=5512)
print(len(freqs))

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
            if freqs[i] % baseFreqs[j] <= 0:
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

print(freqs)



X = librosa.stft(x)
Xdb = (librosa.amplitude_to_db(abs(X)))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()


# print(len(Xdb))
# print(len(Xdb[1]))
# for i in range(len(Xdb)):
#     print(Xdb[20])
# print(Xdb[0][0])

# sample_rate, samples = wavfile.read('/Users/nicholasliu/Documents/adhoncs/soundRecognition/violin-C4.wav')
# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

# print(frequencies)
# print("frequencies:")
# print(len(frequencies))
# print("time:")
# print(len(times))
# print("spectrogram:")
# print(len(spectrogram))
#
# baseFrequency = []
# harmonicList = []
#
# for i in range(len(frequencies)):
#     if freuqencies[i] not in baseFrequency:

# plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# x = np.random.standard_normal(100)
# y = np.random.standard_normal(100)
# z = np.random.standard_normal(100)
# c = np.random.standard_normal(100)
#
# img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
# fig.colorbar(img)
# plt.show()
