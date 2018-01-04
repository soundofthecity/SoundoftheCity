from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
from shogun.Features  import RealFeatures
from shogun.Converter import Jade



def load_wav(filename,samplerate=44100):

    # load file
    rate, data = wavfile.read(filename)

    # convert stereo to mono
    if len(data.shape) > 1:
        data = data[:,0]/2 + data[:,1]/2

    # re-interpolate samplerate    
    #ratio = float(samplerate) / float(rate)
    #data = resample(data, len(data) * ratio)

    return samplerate, data.astype(np.int16)

fs1,s1 = load_wav("s-8/107357.wav")
fs2,s2 = load_wav("d-4/50414.wav")
fs3,s3 = load_wav("sm-9/14387.wav")

fs = fs1
length = max([len(s1), len(s2), len(s3)])

s1.resize((length,1), refcheck=False)
s2.resize((length,1), refcheck=False)
s3.resize((length,1), refcheck=False)

"""
The function numpy.c_ concatenates the numpy arrays given as input.
The method numpy_array.T is the transpose operation that allow us
to prepare an input source matrix of the right size (3, length),
according to the chosen mixing matrix (3,3).
"""
S = (np.c_[s1, s2, s3]).T

# Mixing Matrix
#A = np.random.uniform(size=(3,3))
#A = A / A.sum(axis=0)
A = np.array([[1, 0.5, 0.5],
              [0.5, 1, 0.5],
              [0.5, 0.5, 1]])

A.round(2)

# Mixed Signals
X = np.dot(A,S)

# Exploring Mixed Signals
for i in range(X.shape[0]):
    wavfile.write("mixsounds/mixed_[d,gs,sm]-0.%d" %i,fs,(X[i]).astype(np.int16))



mixed_signals = RealFeatures((X).astype(np.float64))

jade = Jade()
signals = jade.apply(mixed_signals)

S_ = signals.get_feature_matrix()

A_ = jade.get_mixing_matrix()
A_ = A_ / A_.sum(axis=0)

# Convert to features for shogun


gain = 4000
for i in range(S_.shape[0]):
    wavfile.write("sepsounds/seps_[d,gs,sm]-0.%d" %i,fs,(gain*S_[i]).astype(np.int16))

