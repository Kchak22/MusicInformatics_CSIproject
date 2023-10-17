# File for extracting features
import pydub
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
import scipy
import librosa
from copy import deepcopy

#### Feature engineering
def downsampling(features, key, downsample_param=5):
    features_ds = deepcopy(features)
    features_ds[key] = features_ds[key][:,::downsample_param]
    return features_ds

def smoothing(features, key, smoothing_param):
    features_s = deepcopy(features)
    filt_kernel = np.expand_dims(scipy.signal.get_window('hann', smoothing_param), axis=0)
    features_s[key] = scipy.signal.convolve(features[key], filt_kernel, mode='same') / smoothing_param
    return features_s

def normalization(features, key, norm='2'):
    features_normalized = deepcopy(features)
    k, n = features[key].shape
    if norm == '2':
        for i in range(n):
            norm_l2 = np.sqrt(np.sum(features_normalized[key][:,i]**2))
            if norm_l2 < 0.0001:
                features_normalized[key][:,i] = np.ones(k) / np.sqrt(k)
            else: 
                features_normalized[key][:,i] = features_normalized[key][:,i] / norm_l2
    elif norm == '1':
        for i in range(n):
            norm_l1 = np.sum(np.abs(features_normalized[key][:,i]))
            if norm_l1 < 0.0001:
                features_normalized[key][:,i] = np.ones(k) / k
            else: 
                features_normalized[key][:,i] = features_normalized[key][:,i] / norm_l1
    return features_normalized

def quantization(features, key):
    # This function allows to have features more robust to noise
    features_q = deepcopy(features)
    quants = [[0.0, 0.05, 0], [0.05, 0.1, 1], [0.1, 0.2, 2], [0.2, 0.4, 3], [0.4, 1, 4]]
    for lower_bound, upper_bound, quant in quants:
        indices = np.logical_and(lower_bound <= features[key], upper_bound >= features[key])
        features_q[key][indices] = quant
    return features_q

# Our main feature so we wanted to compute it by hand
def compute_chroma(sound, window_size, overlap, fftsize, rate, gamma):
    # STFT computation
    f, t, stft1 = stft(sound, fs=rate, window='hann', nperseg=window_size, noverlap=overlap, nfft=fftsize)
    mag = np.abs(stft1)
    # We start by transforming the frequency axis into a 12-tone resolution axis corresponding to MIDI pitches
    f_borders = 2**((np.arange(129)-69-0.5)/12)*440
    stft_midi = np.zeros((128,mag.shape[1]))
    for i in range(128):
        f_indices = np.logical_and(f>=f_borders[i], f<=f_borders[i+1])
        stft_midi[i,:] = np.sum(mag[f_indices,:],axis=0)
    # Then we sum all notes from the same pitch class
    chromagram = np.zeros((12,mag.shape[1]))
    for i in range(12):
        chromagram[i,:] = np.sum(stft_midi[((np.arange(128)-i)% 12)==0,:], axis=0)
    if gamma > 0:
        return np.log(1 + gamma*chromagram)
    else:
        return chromagram

def compute_cens(features):
    # Section 7.2.1 of the book : CENS features
    features_n = normalization(features, 'chromagram', norm='1')
    features_q = quantization(features_n, 'chromagram')
    features_s = smoothing(features_q, 'chromagram', smoothing_param=40)
    features_ds = downsampling(features_s, 'chromagram', downsample_param=10)
    features_n = normalization(features_ds, 'chromagram')
    return features_n

def compute_tempogram(sound, rate):
    tempogram = librosa.feature.tempogram(y=sound, sr=rate)
    return tempogram

def compute_features(filename, chromagram, mfcc, tempogram, cens):
    # Main function to compute all the features
    # Returns a dictionnary containing the different features we chose
    features = {}
    sound = pydub.AudioSegment.from_file(filename, format='mp3')
    rate = sound.frame_rate
    sound = np.array(sound.get_array_of_samples())
    sound = sound/np.max(sound)
    window_size = int(0.5*rate)
    overlap = window_size - 512
    fftsize = 8192
    n_mfcc = 12
    gamma = 0
    if chromagram:
        features['chromagram'] = compute_chroma(sound, window_size, overlap, fftsize, rate, gamma)
    if cens:
        features['cens'] = compute_cens(features)['chromagram']
    if mfcc:
        features['mfcc'] = librosa.feature.mfcc(y=sound, sr=rate, n_mfcc=n_mfcc, norm='ortho', win_length=window_size, hop_length=window_size-overlap, n_fft=fftsize)
    if tempogram:
        features['tempogram'] = compute_tempogram(sound, rate)
    return features