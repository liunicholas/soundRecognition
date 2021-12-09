# Source: https://pypi.org/project/wavio/
# File: gen_sound.py
# Run
#
#   $ python gen_sound.py
#
# to create the file `sine440.py
#
# Then run (for example on Mac OS)
#
#   $ afplay sine.wav
#
# to listen to the sound
import numpy as np
import wavio
# Parameters
rate = 11025    # samples per second
T = 2           # sample duration (seconds)
f = 440.0       # sound frequency (Hz)
# Compute waveform samples
t = np.linspace(0, T, T*rate, endpoint=False)
x = np.sin(2*np.pi * f * t)
# Write the samples to a file
wavio.write("sine.wav", x, rate, sampwidth=3)
