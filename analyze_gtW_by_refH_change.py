from tqdm import trange
import numpy as np
import torch
import scipy.io as sio

def get_gtW_positive(Ht_real, Ht_imag, Hr_real, Hr_imag, eps = 1e-16):
    assert(Ht_real.size(0) == 2), 'currently, only #mic=2 is supported'

    Ht1_real = Ht_real[0, :]
    Ht1_imag = Ht_imag[0, :]
    Ht2_real = Ht_real[1, :]
    Ht2_imag = Ht_imag[1, :]

    Hr1_real = Hr_real[0, :]
    Hr1_imag = Hr_imag[0, :]
    Hr2_real = Hr_real[1, :]
    Hr2_imag = Hr_imag[1, :]

    # determinant
    det_real = (Ht1_real * Hr2_real - Ht1_imag * Hr2_imag) - (Ht2_real * Hr1_real - Ht2_imag * Hr1_imag)  # FxT
    det_imag = (Ht1_real * Hr2_imag + Ht1_imag * Hr2_real) - (Ht2_real * Hr1_imag + Ht2_imag * Hr1_real)  # FxT

    det_power = det_real * det_real + det_imag * det_imag

    # 1/det
    invdet_real = det_real/ (det_power+eps)
    invdet_imag = -det_imag / (det_power+eps)

    # multiply H (=Wgt)
    Wgt1_real = invdet_real * (Hr2_real-Ht2_real) - invdet_imag * (Hr2_imag-Ht2_imag)
    Wgt1_imag = invdet_real * (Hr2_imag-Ht2_imag) + invdet_imag * (Hr2_real-Ht2_real)
    Wgt2_real = invdet_real * (-Hr1_real+Ht1_real) - invdet_imag * (-Hr1_imag+Ht1_imag)
    Wgt2_imag = invdet_real * (-Hr1_imag+Ht1_imag) + invdet_imag * (-Hr1_real+Ht1_real)

    return torch.cat((Wgt1_real.unsqueeze(0), Wgt2_real.unsqueeze(0)), dim=0), torch.cat((Wgt1_imag.unsqueeze(0), Wgt2_imag.unsqueeze(0)), dim=0)

def get_gtW_negative(Ht_real, Ht_imag, Hr_real, Hr_imag, eps = 1e-16):
    assert(Ht_real.size(0) == 2), 'currently, only #mic=2 is supported'

    Ht1_real = Ht_real[0, :]
    Ht1_imag = Ht_imag[0, :]
    Ht2_real = Ht_real[1, :]
    Ht2_imag = Ht_imag[1, :]

    Hr1_real = Hr_real[0, :]
    Hr1_imag = Hr_imag[0, :]
    Hr2_real = Hr_real[1, :]
    Hr2_imag = Hr_imag[1, :]

    # determinant
    det_real = (Ht1_real * Hr2_real - Ht1_imag * Hr2_imag) - (Ht2_real * Hr1_real - Ht2_imag * Hr1_imag)  # NxFxT
    det_imag = (Ht1_real * Hr2_imag + Ht1_imag * Hr2_real) - (Ht2_real * Hr1_imag + Ht2_imag * Hr1_real)  # NxFxT

    det_power = det_real * det_real + det_imag * det_imag

    # 1/det
    invdet_real = det_real/ (det_power+eps)
    invdet_imag = -det_imag / (det_power+eps)

    # multiply H (=Wgt)
    Wgt1_real = invdet_real * Hr2_real - invdet_imag * Hr2_imag
    Wgt1_imag = invdet_real * Hr2_imag + invdet_imag * Hr2_real
    Wgt2_real = invdet_real * (-Hr1_real) - invdet_imag *(-Hr1_imag)
    Wgt2_imag = invdet_real * (-Hr1_imag) + invdet_imag * (-Hr1_real)

    return torch.cat((Wgt1_real.unsqueeze(0), Wgt2_real.unsqueeze(0)), dim=0), torch.cat((Wgt1_imag.unsqueeze(0), Wgt2_imag.unsqueeze(0)), dim=0)

# tarIR manifest
tarIR_manifest_path = 'data_sorted/L553_30301_srcfree_ref1_ofa_RT0.2.csv'

# refer IR list (ref1*2 + ref2*2)
refIR_prefix = '/data4/kenkim/RIRs_L553_smallset_val_te_ref'
refIR_list = [refIR_prefix + '/ref1/s=0.718044916802_4.16879616772_2.04637468378_RT0.2',
              refIR_prefix + '/ref1/s=0.998054530018_4.24651247678_2.14280828293_RT0.2',
              refIR_prefix + '/ref2/s=4.00029335599_0.879268346963_0.90229746076_RT0.2',
              refIR_prefix + '/ref2/s=4.29979040094_0.754075360682_0.834079060442_RT0.2'] # note that first refIR is used for training of ofa
nRef = len(refIR_list)

for r in range(nRef):
    print('refIR = ' + refIR_list[r])
    refIR_ch1 = np.squeeze(np.load(refIR_list[r] + '_ch1.npy'))
    refIR_ch2 = np.squeeze(np.load(refIR_list[r] + '_ch2.npy'))
    refIR_nMic = torch.FloatTensor(np.stack([refIR_ch1, refIR_ch2])).squeeze() # MxT
    refIR_realimag = torch.FloatTensor(refIR_nMic.size(0), refIR_nMic.size(1), 2).zero_() # MxTx2
    refIR_realimag[:, :, 0] = refIR_nMic
    refH = torch.fft(refIR_realimag.cuda(), signal_ndim=1) # MxFx2
    refH = refH[:, :int(refH.size(1)/2+1),:]

    # target IR list
    manifest_r = open(tarIR_manifest_path, 'r')
    lines = manifest_r.readlines()
    nLine = len(lines)
    for t in trange(0, nLine):
        # read tarIR
        line = lines[t].strip()
        line_splited = line.split(',')
        tarIR_prefix = line_splited[0]
        tarIR_ch1 = np.squeeze(np.load(tarIR_prefix + '_ch1.npy'))
        tarIR_ch2 = np.squeeze(np.load(tarIR_prefix + '_ch2.npy'))
        tarIR_nMic = torch.FloatTensor(np.stack([tarIR_ch1, tarIR_ch2])).squeeze() # MxT
        tarIR_realimag = torch.FloatTensor(tarIR_nMic.size(0), tarIR_nMic.size(1), 2).zero_()  # MxTx2
        tarIR_realimag[:, :, 0] = tarIR_nMic
        tarH = torch.fft(tarIR_realimag.cuda(), signal_ndim=1) # MxFx2
        tarH = tarH[:, :int(tarH.size(1) / 2 + 1), :]

        # get Wgt by negative refIR
        Wgt_N_real, Wgt_N_imag = get_gtW_negative(tarH[..., 0], tarH[..., 1], refH[..., 0], refH[..., 1]) # (MxF), (MxF)

        # get Wgt by positive refIR
        Wgt_P_real, Wgt_P_imag = get_gtW_positive(tarH[..., 0], tarH[..., 1], refH[..., 0], refH[..., 1]) # (MxF), (MxF)

        # save W & infos
        savename = 'gtW_by_refH/tar=' + str(t) + ', ref=' + str(r) + '.mat'
        sio.savemat(savename, {'tarIR_info':tarIR_prefix, 'refIR_info':refIR_list[r],
                               'Wgt_N_real':Wgt_N_real.data.cpu().numpy(), 'Wgt_N_imag':Wgt_N_imag.data.cpu().numpy(),
                               'Wgt_P_real': Wgt_P_real.data.cpu().numpy(), 'Wgt_P_imag': Wgt_P_imag.data.cpu().numpy()})

    manifest_r.close()