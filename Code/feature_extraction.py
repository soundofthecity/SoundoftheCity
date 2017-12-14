import numpy as np
import librosa
import os


def extract_features(raw_sound, sample_rate):
    stft = np.abs(librosa.stft(raw_sound))
    # Mel-frequency cepstral coefficients
    mfcc = np.mean(librosa.feature.mfcc(y=raw_sound, sr=sample_rate, n_mfcc=40).T, axis=0)
    # Chromagram from a waveform or power spectrogram
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    #  Mel-scaled power spectrogram
    mel = np.mean(librosa.feature.melspectrogram(raw_sound, sr=sample_rate).T, axis=0)
    # Spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    # The tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(raw_sound), sr=sample_rate).T, axis=0)

    return mfcc, chroma, mel, contrast, tonnetz


sample_rates = np.load("sample_rates.npy")
labels = np.load("raw_sounds_id.npy")

s1 = np.load("raw_data/0.npy")
m, c, mel, cn, tn = extract_features(s1, sample_rates[0])
numsample = len(sample_rates)

mfccs = np.empty((numsample, m.shape[0]))
chromas = np.empty((numsample, c.shape[0]))
mels = np.empty((numsample, mel.shape[0]))
contrasts = np.empty((numsample, cn.shape[0]))
tonnets = np.empty((numsample, tn.shape[0]))


def take_raw_sounds(path):
    index = 0

    for raw_sound in range(8072):
        raw = np.load(path + str(raw_sound) + '.npy')

        try:
            mx, cx, melx, cnx, tnx = extract_features(raw, sample_rates[index])
            mfccs[index] = mx
            chromas[index] = cx
            mels[index] = melx
            contrasts[index] = cnx
            tonnets[index] = tnx
        except:
            print(path, raw_sound, ".npy")

        index += 1
        print(index)


take_raw_sounds("raw_data/")

np.save("mfccs.npy", mfccs)
np.save("chromas.npy", chromas)
np.save("mels.npy", mels)
np.save("contrasts.npy", contrasts)
np.save("tonnetz.npy", tonnets)
