import scipy.io as sio
from models.layers.istft import ISTFT
import torch
import soundfile as sf
import os
import pickle
import numpy as np
from scipy.signal import fftconvolve
import librosa
import pdb

def normalize(y):
    y = y/max(abs(y))
    return y

# Based on https://github.com/librosa/librosa/issues/434
def _stft(y, n_fft, hop_len):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_len)

def _istft(y, hop_len):
    return librosa.istft(y, hop_length=hop_len)

def _griffin_lim(S, n_fft, hop_len):
    griffin_lim_iters = 60
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    for i in range(griffin_lim_iters):
        if i > 0:
            angles = np.exp(1j * np.angle(_stft(y, n_fft, hop_len)))
        y = _istft(S_complex * angles, hop_len)
    return y


# define ISTFT
n_fft = 8192
#n_fft = 128
win_size = n_fft
hop_length = int(win_size/2)
eps = 1e-12

savename_ISTFT = 'ISTFT_' + str(n_fft) + '.pth'
if not os.path.exists(savename_ISTFT):
    print('init ISTFT')
    istft = ISTFT(n_fft, hop_length, window='hanning')
    with open(savename_ISTFT, 'wb') as f:
        pickle.dump(istft, f)
else:
    print('load saved ISTFT')
    with open(savename_ISTFT, 'rb') as f:
        istft = pickle.load(f)


# load samples from spec
expnum = 355 # 1cm train, 0.1cm test, 16L
if not os.path.exists('wavs/' + str(expnum)):
    os.makedirs('wavs/' + str(expnum))

print('load samples')
samples = sio.loadmat('specs/' + str(expnum) + '/tr_0.mat')
meta = sio.loadmat('specs/' + str(expnum) + '/SDR_tr_1.mat')

clean_imag = torch.FloatTensor(samples['clean_imag'])
clean_real = torch.FloatTensor(samples['clean_real'])
mixed_real = torch.FloatTensor(samples['mixed_real'])
mixed_imag = torch.FloatTensor(samples['mixed_imag'])
out_real = torch.FloatTensor(samples['out_real'])
out_imag = torch.FloatTensor(samples['out_imag'])

# check ISTFT
'''
print('ISTFT of clean, mixed')
clean_audio_s0 = istft(clean_real[0], clean_imag[0], length=None)
clean_audio_s1 = istft(clean_real[1], clean_imag[1], length=None)
mixed_audio_s0_m0 = istft(mixed_real[0][0], mixed_imag[0][0], length=None)
mixed_audio_s0_m1 = istft(mixed_real[0][1], mixed_imag[0][1], length=None)
mixed_audio_s1_m0 = istft(mixed_real[1][0], mixed_imag[1][0], length=None)
mixed_audio_s1_m1 = istft(mixed_real[1][1], mixed_imag[1][1], length=None)

# generate reverb by convolution
reverb_path = meta['reverb_path']
h_s0_m0 = np.squeeze(np.load(reverb_path[0] + '_ch1.npy'))
h_s0_m1 = np.squeeze(np.load(reverb_path[0] + '_ch2.npy'))
h_s1_m0 = np.squeeze(np.load(reverb_path[1] + '_ch1.npy'))
h_s1_m1 = np.squeeze(np.load(reverb_path[1] + '_ch2.npy'))

tau1_s0 = np.argmax(h_s0_m0)
tau1_s1 = np.argmax(h_s1_m0)

x_s0_m0 = fftconvolve(clean_audio_s0.numpy(), h_s0_m0)
x_s0_m1 = fftconvolve(clean_audio_s0.numpy(), h_s0_m1)
x_s1_m0 = fftconvolve(clean_audio_s1.numpy(), h_s1_m0)
x_s1_m1 = fftconvolve(clean_audio_s1.numpy(), h_s1_m1)

# save wav


sf.write('wavs/' + str(expnum) + '/clean_s0.wav', normalize(clean_audio_s0).data.numpy(), 16000)
sf.write('wavs/' + str(expnum) + '/clean_s1.wav', normalize(clean_audio_s1).data.numpy(), 16000)

sf.write('wavs/' + str(expnum) + '/x_s0_m0_recon.wav', normalize(mixed_audio_s0_m0).data.numpy(), 16000)
sf.write('wavs/' + str(expnum) + '/x_s0_m1_recon.wav', normalize(mixed_audio_s0_m1).data.numpy(), 16000)
sf.write('wavs/' + str(expnum) + '/x_s1_m0_recon.wav', normalize(mixed_audio_s1_m0).data.numpy(), 16000)
sf.write('wavs/' + str(expnum) + '/x_s1_m1_recon.wav', normalize(mixed_audio_s1_m1).data.numpy(), 16000)

sf.write('wavs/' + str(expnum) + '/x_s0_m0.wav', normalize(x_s0_m0), 16000)
sf.write('wavs/' + str(expnum) + '/x_s0_m1.wav', normalize(x_s0_m1), 16000)
sf.write('wavs/' + str(expnum) + '/x_s1_m0.wav', normalize(x_s1_m0), 16000)
sf.write('wavs/' + str(expnum) + '/x_s1_m1.wav', normalize(x_s1_m1), 16000)

x_s0_m0 = x_s0_m0[tau1_s0:]
x_s0_m1 = x_s0_m1[tau1_s0:]
x_s1_m0 = x_s1_m0[tau1_s1:]
x_s1_m1 = x_s1_m1[tau1_s1:]

sf.write('wavs/' + str(expnum) + '/x_s0_m0_tau1.wav', normalize(x_s0_m0), 16000)
sf.write('wavs/' + str(expnum) + '/x_s0_m1_tau1.wav', normalize(x_s0_m1), 16000)
sf.write('wavs/' + str(expnum) + '/x_s1_m0_tau1.wav', normalize(x_s1_m0), 16000)
sf.write('wavs/' + str(expnum) + '/x_s1_m1_tau1.wav', normalize(x_s1_m1), 16000)
'''

# Magnitude estimation + wav generation (method 1. use clean phase) --> 결론: 쓰지말자
# given: out_mag, clean_real, clean_imag
# DO NOT USE out_real, out_imag itself
clean_mag = torch.sqrt(clean_real*clean_real + clean_imag*clean_imag)
out_mag = torch.sqrt(out_real*out_real + out_imag*out_imag)

'''
ratio = out_mag/(clean_mag+eps)

out_real_cleanphs = clean_real*ratio
out_imag_cleanphs = clean_imag*ratio

#pdb.set_trace()
out_audio_s0_cleanphs = istft(out_real_cleanphs[0], out_imag_cleanphs[0], length=None)
out_audio_s1_cleanphs = istft(out_real_cleanphs[1], out_imag_cleanphs[1], length=None)

sio.savemat('gt_gen.mat', {'clean_mag':clean_mag.data.numpy(), 'out_mag':out_mag.data.numpy(),
                           'ratio':ratio.data.numpy(),
                           'out_audio_s0_cleanphs':out_audio_s0_cleanphs.data.numpy(),
                           'out_audio_s1_cleanphs':out_audio_s1_cleanphs.data.numpy()})

sf.write('wavs/' + str(expnum) + '/out_s0_cleanphs.wav', normalize(out_audio_s0_cleanphs).data.numpy(), 16000)
sf.write('wavs/' + str(expnum) + '/out_s1_cleanphs.wav', normalize(out_audio_s1_cleanphs).data.numpy(), 16000)
'''

# Magnitude est. + wav generation (method 2. Griffin-Lim)

# win_size = 8192
s0 = clean_mag[0, :, :].numpy()
o0 = out_mag[0, :, :].numpy()
S_s0_GL = _griffin_lim(s0**1.5, n_fft=n_fft, hop_len=hop_length)
O_s0_GL = _griffin_lim(o0**1.5, n_fft=n_fft, hop_len=hop_length)

sf.write('wavs/' + str(expnum) + '/s0_GL.wav', normalize(S_s0_GL), 16000)
sf.write('wavs/' + str(expnum) + '/o0_GL.wav', normalize(O_s0_GL), 16000)

# win_size = 512