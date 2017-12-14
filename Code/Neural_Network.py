from sklearn.neural_network import MLPClassifier
import numpy as np
import random

mfcc = np.load("mfccs.npy")
labels = np.load("raw_sounds_id.npy")

a = list(zip(mfcc, labels))
random.shuffle(a)
mfcc, labels = zip(*a)

mlp = MLPClassifier(activation='logistic', batch_size=16, hidden_layer_sizes=(250, 10), learning_rate_init=0.002)

mlp.fit(mfcc[:8000], labels[:8000])
print((mlp.score(mfcc[8000:], labels[8000:])))

