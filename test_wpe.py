import numpy as np
import soundfile as sf
import os
from scipy.signal import fftconvolve
import pdb

def normalize(y):
    y = y/max(abs(y))
    return y

# Make reverberant audio (700ms)
sample_name = '7001-12337-0104'
h1 = np.squeeze(np.load('s=4.31_4.4_1.35_RT0.7_ch1.npy'))
h2 = np.squeeze(np.load('s=4.31_4.4_1.35_RT0.7_ch2.npy'))

x1_path = '/data2/kenkim/' + sample_name + '_ch1_RT0.7.wav'
x2_path = '/data2/kenkim/' + sample_name + '_ch2_RT0.7.wav'

x1_path_normalized = '/data2/kenkim/' + sample_name + '_ch1_RT0.7_normalized.wav'
x2_path_normalized = '/data2/kenkim/' + sample_name + '_ch2_RT0.7_normalized.wav'

if os.path.exists(x1_path) and os.path.exists(x2_path):
    x1, fs = sf.read(x1_path)
    x2, fs = sf.read(x2_path)
else:
    s, fs = sf.read('/data2/kenkim/' + sample_name + '.wav')
    assert(fs == 16000) # h assume fs = 16000
    x1 = fftconvolve(s, h1)
    sf.write(x1_path, x1, fs)
    x2 = fftconvolve(s, h2)
    sf.write(x2_path, x2, fs)

    sf.write(x1_path_normalized, normalize(x1), fs)
    sf.write(x2_path_normalized, normalize(x2), fs)

x = np.stack([x1, x2], axis=0)

# M1: Nara-WPE
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft

stft_options = dict(size=512, shift=128)
#pdb.set_trace()
X = stft(x, **stft_options)
X = X.transpose(2, 0, 1)
Z = wpe(X).transpose(1, 2, 0)
z = istft(Z, **stft_options)
#pdb.set_trace()
sf.write('nara_wpe_' + sample_name + '_1.wav', normalize(z[0]), fs)
sf.write('nara_wpe_' + sample_name + '_2.wav', normalize(z[1]), fs)