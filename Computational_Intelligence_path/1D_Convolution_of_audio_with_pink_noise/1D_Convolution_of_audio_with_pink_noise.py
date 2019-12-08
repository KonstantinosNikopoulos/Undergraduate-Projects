#!/usr/bin/env python
# coding: utf-8

#  1D convolution of audio with pink noise (wav files)

#  author: Konstantinos Nikopoulos

from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt


# 1D Convolution
def MyConvolve(A,B):
    length = len(A) + len(B) -1 # length of result
    A_pad = np.pad(A, (length-len(A), 0), 'constant', constant_values=(0)) # pads zeros in front of A
    B_pad = np.pad(B, (length-len(B), 0), 'constant', constant_values=(0)) # pads zeros in front of B
    B_pad = B_pad[::-1] # reverses B_pad
    C = np.zeros(length) 
    for i in range(0,length):
        sum = 0
        for j in range(0,length):
            sum = sum + A_pad[j] * B_pad[j-i]
        C[i] = sum
    return C;

# Scaling to [tmin,tmax]
def scaling(signal,tmin,tmax):
    signal_n = np.zeros(len(signal))
    for i in range(0,len(signal)):
        signal_n[i] = (tmax - tmin) * (signal[i] - min(signal))/(max(signal) - min(signal)) + tmin
    return signal_n

# Read wav files
fs_1, sample_audio = wavfile.read('./audio/sample_audio.wav')
fs_2, pink_noise = wavfile.read('./audio/pink_noise.wav')

# Scaling to -1 1
sample_audio_n = scaling(sample_audio,-1,1)
pink_noise_n = scaling(pink_noise,-1,1)

# Convolution of sample_audio and pink_noise
pinkNoise_sampleAudio_n = MyConvolve(sample_audio_n,pink_noise_n) 

# Scaling result to sample's range
pinkNoise_sampleAudio = scaling(pinkNoise_sampleAudio_n,min(sample_audio),max(sample_audio))

# Plot result
plt.figure(figsize=(10, 2))
plt.plot(pinkNoise_sampleAudio, color='gray')
plt.xlim([0, pinkNoise_sampleAudio.shape[0]])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Write result of convolution in wav file
wavfile.write('./audio/pinkNoise_sampleAudio.wav',fs_1,pinkNoise_sampleAudio)




