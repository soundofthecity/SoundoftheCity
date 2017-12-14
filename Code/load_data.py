import librosa
import csv
import numpy as np
import librosa.display


def load_data(path):
    class_Ids = []
    sample_rates = []
    x = 0
    with open(path, 'r') as csvfile:
        metadata = csv.DictReader(csvfile)
        for row in metadata:
            soundDir = "UrbanSound8K/audio/fold" + row['fold'] + "/" + row['slice_file_name']
            soundClassId = row['classID']
            print(x)
            raw_sound, sample_rate = librosa.load(soundDir)
            class_Ids.append(int(soundClassId))
            sample_rates.append(sample_rate)
            np.save("raw_data/" + str(x) + ".npy", raw_sound)
            x += 1

    class_Ids = np.array(class_Ids)
    sample_rates = np.array(sample_rates)


    np.save("raw_sounds_id.npy", class_Ids)
    np.save("sample_rates.npy", sample_rates)

load_data("UrbanSound8K/metadata/UrbanSound8K.csv")

