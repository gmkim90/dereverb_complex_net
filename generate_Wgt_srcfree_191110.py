from tqdm import trange
import numpy as np
import torch
import scipy.io as sio
import pdb

def get_gtW_negative(Ht1_real, Ht1_imag, Ht2_real, Ht2_imag, Hr1_real, Hr1_imag, Hr2_real, Hr2_imag, eps = 1e-16):
    #assert(Ht1_real.size(1) == 2), 'currently, only #mic=2 is supported'

    # determinant
    det_real = (Ht1_real * Hr2_real - Ht1_imag * Hr2_imag) - (Ht2_real * Hr1_real - Ht2_imag * Hr1_imag)  # NxFx1
    det_imag = (Ht1_real * Hr2_imag + Ht1_imag * Hr2_real) - (Ht2_real * Hr1_imag + Ht2_imag * Hr1_real)  # NxFx1

    det_power = det_real * det_real + det_imag * det_imag # NxFx1

    # 1/det
    invdet_real = det_real/ (det_power+eps) # NxFx1
    invdet_imag = -det_imag / (det_power+eps) # NxFx1

    # multiply H (=Wgt)
    Wgt1_real = invdet_real * Hr2_real - invdet_imag * Hr2_imag # NxFx1
    Wgt1_imag = invdet_real * Hr2_imag + invdet_imag * Hr2_real # NxFx1
    Wgt2_real = invdet_real * (-Hr1_real) - invdet_imag *(-Hr1_imag) # NxFx1
    Wgt2_imag = invdet_real * (-Hr1_imag) + invdet_imag * (-Hr1_real) # NxFx1

    return Wgt1_real, Wgt1_imag, Wgt2_real, Wgt2_imag


# src free
test_manifest = open('data_sorted/L553_30301_1_unseenSrc1_ref1_ofa.csv', 'r')
lines = test_manifest.readlines()

refIR_ch1 = None
refIR_ch2 = None

nFFT = 8192 # to compare with src-dependent result
#nFFT = 3200

#eps = 1e-16
eps = 1e-6



nData = len(lines) # position = 961

W1_tot = torch.zeros(nData, nFFT, 2)
W2_tot = torch.zeros(nData, nFFT, 2)

M = 2

Ht1_real = torch.zeros(nData, nFFT)
Ht1_imag = torch.zeros(nData, nFFT)
Ht2_real = torch.zeros(nData, nFFT)
Ht2_imag = torch.zeros(nData, nFFT)
#Hr1_real = torch.zeros(nData, nFFT)
#Hr1_imag = torch.zeros(nData, nFFT)
#Hr2_real = torch.zeros(nData, nFFT)
#Hr2_imag = torch.zeros(nData, nFFT)

for i in trange(0, nData):
    line = lines[i]
    line_splited = line.split(',')
    tarIR_path = line_splited[0]
    refIR_path = line_splited[2].strip()

    tarIR_ch1 = np.squeeze(np.load(tarIR_path + '_ch1.npy'))
    tarIR_ch2 = np.squeeze(np.load(tarIR_path + '_ch2.npy'))

    tarIR_complex = torch.FloatTensor(M, nFFT, 2).zero_()
    tarIR_complex[0, :tarIR_ch1.shape[0], 0] = torch.FloatTensor(tarIR_ch1) # assign real part, imaginary part = 0
    tarIR_complex[1, :tarIR_ch2.shape[0], 0] = torch.FloatTensor(tarIR_ch2) # assign real part, imaginary part = 0
    #tarIR_complex = tarIR_complex.cuda()

    Ht = torch.fft(tarIR_complex, signal_ndim=1) # MxTx2 --> MxFx2
    #Ht = Ht.cpu()

    Ht1_real_i, Ht1_imag_i = Ht[0, :, 0], Ht[0, :, 1]
    Ht2_real_i, Ht2_imag_i = Ht[1, :, 0], Ht[1, :, 1]

    Ht1_real[i, ...] = Ht1_real_i
    Ht1_imag[i, ...] = Ht1_imag_i
    Ht2_real[i, ...] = Ht2_real_i
    Ht2_imag[i, ...] = Ht2_imag_i

    if(refIR_ch2 is None): # same for all test data
        refIR_ch1 = np.squeeze(np.load(refIR_path + '_ch1.npy'))
        refIR_ch2 = np.squeeze(np.load(refIR_path + '_ch2.npy'))

        refIR_complex = torch.FloatTensor(M, nFFT, 2).zero_()
        refIR_complex[0, :refIR_ch1.shape[0], 0] = torch.FloatTensor(refIR_ch1)  # assign real part, imaginary part = 0
        refIR_complex[1, :refIR_ch2.shape[0], 0] = torch.FloatTensor(refIR_ch2)  # assign real part, imaginary part = 0

        Hr = torch.fft(refIR_complex, signal_ndim=1)  # MxTx2 --> MxFx2

        Hr1_real_i, Hr1_imag_i = Hr[0, :, 0], Hr[0, :, 1]
        Hr2_real_i, Hr2_imag_i = Hr[1, :, 0], Hr[1, :, 1]

        Hr1_real = Hr1_real_i.view(1, Hr1_real_i.size(0)).expand(Ht1_real.size())
        Hr1_imag = Hr1_imag_i.view(1, Hr1_imag_i.size(0)).expand(Ht1_imag.size())
        Hr2_real = Hr2_real_i.view(1, Hr2_real_i.size(0)).expand(Ht2_real.size())
        Hr2_imag = Hr2_imag_i.view(1, Hr2_imag_i.size(0)).expand(Ht2_imag.size())

Wgt1_real, Wgt1_imag, Wgt2_real, Wgt2_imag = \
    get_gtW_negative(Ht1_real, Ht1_imag, Ht2_real, Ht2_imag, Hr1_real, Hr1_imag, Hr2_real, Hr2_imag, eps=eps)

#pdb.set_trace()
W1_tot[:, :, 0] = Wgt1_real
W1_tot[:, :, 1] = Wgt1_imag
W2_tot[:, :, 0] = Wgt2_real
W2_tot[:, :, 1] = Wgt2_imag

sio.savemat('Wgt_srcfree_nFFT' + str(nFFT) + 'eps=' + str(eps) + '.mat', {'W1_tot':W1_tot.data.numpy(), 'W2_tot':W2_tot.data.numpy()})



test_manifest.close()