import glob
import os
import librosa
import numpy as np
import librosa.display



def load_sounds(path):
    raw_sounds = []
    sound_labels = []
    sample_rates = []
    for label in os.listdir(path):
        subdir = (path + "/" + label)

        sound_wav = glob.glob(subdir + "/" + '*.wav')
        sound_mp3 = glob.glob(subdir + "/" + '*.mp3')
        sounds = sound_mp3 + sound_wav
        for sound in sounds:
            print(sound)
            try:
                x, sample_rate = librosa.load(sound)
                raw_sounds.append(x)
                sample_rates.append(sample_rate)
                sound_labels.append(label[1])
            except:
                continue

    return raw_sounds, sample_rates,sound_labels


rs, sr, sl = load_sounds("sounds")
np.save("sample_rates.npy", sr)
np.save("soundlabels.npy", sl)

output_raw = "raw_sounds/"
for raw_ind in range(len(rs)):
    print(raw_ind)
    np.save(output_raw+str(raw_ind)+".npy", np.array(rs[raw_ind]))




