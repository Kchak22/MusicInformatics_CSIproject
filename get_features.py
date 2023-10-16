# File for extracting features
import pydub
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
import librosa

def compute_features(filename, chromagram, mfcc, tempogram):
    features = {}
    sound = pydub.AudioSegment.from_file(filename, format='mp3')
    rate = sound.frame_rate
    sound = np.array(sound.get_array_of_samples())
    sound = sound/np.max(sound)
    window_size = int(100*1e-3*rate)
    overlap = int(90*1e-3*rate)
    fftsize = 8192
    n_mfcc = 12
    gamma = 100
    if chromagram:
        features['chromagram'] = compute_chroma(sound, window_size, overlap, fftsize, rate, gamma)
    if mfcc:
        features['mfcc'] = compute_mfcc(sound, rate, n_mfcc, window_size, window_size-overlap)
    if tempogram:
        features['tempogram'] = compute_tempogram(sound, rate)
    return features


# Our main feature so we wanted to compute it by hand
def compute_chroma(sound, window_size, overlap, fftsize, rate, gamma):
    # STFT computation
    f, t, stft1 = stft(sound, fs=rate, window='hann', nperseg=window_size, noverlap=overlap)
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
    chromagram_log = np.log(1 + gamma*chromagram)
    # plt.pcolormesh(t, np.arange(12), chromagram_log, cmap='cividis')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (kHz)')
    # plt.show()
    return chromagram_log

def compute_hpcp(sound):
    return

def compute_mfcc(sound, rate, n_mfcc, window_size, window_hop):
    mfcc = librosa.feature.mfcc(y=sound, sr=rate, n_mfcc=n_mfcc, norm='ortho', win_length=window_size, hop_length=window_hop)
    # librosa.display.specshow(mfcc, sr=rate, x_axis='time')
    # plt.title('MFCC')
    # plt.ylabel('DCT coeff')
    # plt.colorbar()
    return mfcc

def compute_tempogram(sound, rate):
    tempogram = librosa.feature.tempogram(y=sound, sr=rate)
    # librosa.display.specshow(tempogram, sr=rate, x_axis='time')
    # plt.ylim(80,200)
    # plt.ylabel('Beats (BPM)')
    # plt.colorbar()
    return tempogram