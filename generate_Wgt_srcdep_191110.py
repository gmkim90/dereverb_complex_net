from tqdm import trange
import numpy as np
import torch
import scipy.io as sio
import soundfile as sf
from scipy.signal import fftconvolve
import pdb

def get_gtW_negative(Xt1_real, Xt1_imag, Xt2_real, Xt2_imag, Xr1_real, Xr1_imag, Xr2_real, Xr2_imag, S_real, S_imag, eps = 1e-16):
    #assert(Ht1_real.size(1) == 2), 'currently, only #mic=2 is supported'

    # determinant
    det_real = (Xt1_real * Xr2_real - Xt1_imag * Xr2_imag) - (Xt2_real * Xr1_real - Xt2_imag * Xr1_imag)  # NxFx1
    det_imag = (Xt1_real * Xr2_imag + Xt1_imag * Xr2_real) - (Xt2_real * Xr1_imag + Xt2_imag * Xr1_real)  # NxFx1

    det_power = det_real * det_real + det_imag * det_imag # NxFx1

    # S/det
    S_det_real = (S_real * det_real + S_imag * det_imag)/ (det_power+eps) # NxFxT
    S_det_imag = (S_imag * det_real - S_real * det_imag)/ (det_power+eps) # NxFxT

    # multiply Xref (=Wgt)
    Wgt1_real = S_det_real * Xr2_real - S_det_imag * Xr2_imag
    Wgt1_imag = S_det_real * Xr2_imag + S_det_imag * Xr2_real
    Wgt2_real = S_det_real * (-Xr1_real) - S_det_imag * (-Xr1_imag)
    Wgt2_imag = S_det_real * (-Xr1_imag) + S_det_imag * (-Xr1_real)

    return Wgt1_real, Wgt1_imag, Wgt2_real, Wgt2_imag

# src free
test_manifest = open('data_sorted/L553_30301_1_unseenSrc1_ref1_ofa.csv', 'r')
lines = test_manifest.readlines()

refIR_ch1 = None
refIR_ch2 = None

nFFT = 8192 # to compare with src-dependent result
F = int(nFFT/2+1)
#nFFT = 3200
hop_length = int(nFFT/2)
nWin = nFFT
window_path = 'window_' + str(nWin) + '.pth'
window = torch.load(window_path, map_location=torch.device('cpu'))

stft = lambda x: torch.stft(torch.FloatTensor(x), nFFT, hop_length, win_length=nWin, window=window)

eps = 1e-16
#eps = 1e-6

nData = len(lines) # position = 961

W1_tot = torch.zeros(nData, F, 2)
W2_tot = torch.zeros(nData, F, 2)

M = 2

src_path = '/home/kenkim/librispeech_kaldi/LibriSpeech/train/wav/119-129515-0016.wav'
src, sr = sf.read(src_path)

S = stft(src)
S_real, S_imag = S[..., 0], S[..., 1]

T = S_real.size(-1)


Xt1_real = torch.zeros(nData, F, T)
Xt1_imag = torch.zeros(nData, F, T)
Xt2_real = torch.zeros(nData, F, T)
Xt2_imag = torch.zeros(nData, F, T)
#Xr1_real = torch.zeros(nData, nFFT)
#Xr1_imag = torch.zeros(nData, nFFT)
#Xr2_real = torch.zeros(nData, nFFT)
#Xr2_imag = torch.zeros(nData, nFFT)

for i in trange(0, nData):
    line = lines[i]
    line_splited = line.split(',')
    tarIR_path = line_splited[0]
    refIR_path = line_splited[2].strip()

    tarIR_ch1 = np.squeeze(np.load(tarIR_path + '_ch1.npy'))
    tarIR_ch2 = np.squeeze(np.load(tarIR_path + '_ch2.npy'))

    xt1 = fftconvolve(src, tarIR_ch1)
    xt2 = fftconvolve(src, tarIR_ch2)

    Xt1 = stft(xt1)
    Xt2 = stft(xt2)

    Xt1 = Xt1[:, :T, :]
    Xt2 = Xt2[:, :T, :]

    Xt1_real[i, :, :] = Xt1[..., 0]
    Xt1_imag[i, :, :] = Xt1[..., 1]
    Xt2_real[i, :, :] = Xt2[..., 0]
    Xt2_imag[i, :, :] = Xt2[..., 1]

    if(refIR_ch2 is None): # same for all test data
        refIR_ch1 = np.squeeze(np.load(refIR_path + '_ch1.npy'))
        refIR_ch2 = np.squeeze(np.load(refIR_path + '_ch2.npy'))

        xr1 = fftconvolve(src, refIR_ch1)
        xr2 = fftconvolve(src, refIR_ch2)

        Xr1 = stft(xr1)
        Xr2 = stft(xr2)

        Xr1 = Xr1[:, :T, :]
        Xr2 = Xr2[:, :T, :]

        Xr1_real_i, Xr1_imag_i = Xr1[..., 0], Xr1[..., 1]
        Xr2_real_i, Xr2_imag_i = Xr2[..., 0], Xr2[..., 1]

        Xr1_real = Xr1_real_i.view(1, Xr1_real_i.size(0), T).expand(Xt1_real.size())
        Xr1_imag = Xr1_imag_i.view(1, Xr1_imag_i.size(0), T).expand(Xt1_imag.size())
        Xr2_real = Xr2_real_i.view(1, Xr2_real_i.size(0), T).expand(Xt2_real.size())
        Xr2_imag = Xr2_imag_i.view(1, Xr2_imag_i.size(0), T).expand(Xt2_imag.size())

Wgt1_real, Wgt1_imag, Wgt2_real, Wgt2_imag = \
    get_gtW_negative(Xt1_real, Xt1_imag, Xt2_real, Xt2_imag, Xr1_real, Xr1_imag, Xr2_real, Xr2_imag, S_real, S_imag, eps=eps)

#pdb.set_trace()
W1_tot[:, :, 0] = Wgt1_real.mean(2) # temporal mean
W1_tot[:, :, 1] = Wgt1_imag.mean(2)
W2_tot[:, :, 0] = Wgt2_real.mean(2)
W2_tot[:, :, 1] = Wgt2_imag.mean(2)

sio.savemat('Wgt_srcdep_nFFT' + str(nFFT) + 'eps=' + str(eps) + '.mat', {'W1_tot':W1_tot.data.numpy(), 'W2_tot':W2_tot.data.numpy()})



test_manifest.close()