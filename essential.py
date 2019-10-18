import os

import torch
import torch.nn.functional as Func
#import librosa
#import torchaudio
import soundfile as sf
import math
import scipy.io as sio
from models.loss import var_time

import pdb

def get_gtW_positive(Ht_real, Ht_imag, Hr_real, Hr_imag, eps = 1e-16):
    assert(Ht_real.size(1) == 2), 'currently, only #mic=2 is supported'

    Ht1_real = Ht_real[:, 0, :, :]
    Ht1_imag = Ht_imag[:, 0, :, :]
    Ht2_real = Ht_real[:, 1, :, :]
    Ht2_imag = Ht_imag[:, 1, :, :]

    Hr1_real = Hr_real[:, 0, :, :]
    Hr1_imag = Hr_imag[:, 0, :, :]
    Hr2_real = Hr_real[:, 1, :, :]
    Hr2_imag = Hr_imag[:, 1, :, :]

    # determinant
    det_real = (Ht1_real * Hr2_real - Ht1_imag * Hr2_imag) - (Ht2_real * Hr1_real - Ht2_imag * Hr1_imag)  # NxFxT
    det_imag = (Ht1_real * Hr2_imag + Ht1_imag * Hr2_real) - (Ht2_real * Hr1_imag + Ht2_imag * Hr1_real)  # NxFxT

    det_power = det_real * det_real + det_imag * det_imag

    # 1/det
    invdet_real = det_real/ (det_power+eps)
    invdet_imag = -det_imag / (det_power+eps)

    # multiply H (=Wgt)
    Wgt1_real = invdet_real * (Hr2_real-Ht2_real) - invdet_imag * (Hr2_imag-Ht2_imag)
    Wgt1_imag = invdet_real * (Hr2_imag-Ht2_imag) + invdet_imag * (Hr2_real-Ht2_real)
    Wgt2_real = invdet_real * (-Hr1_real+Ht1_real) - invdet_imag * (-Hr1_imag+Ht1_imag)
    Wgt2_imag = invdet_real * (-Hr1_imag+Ht1_imag) + invdet_imag * (-Hr1_real+Ht1_real)

    return torch.cat((Wgt1_real.unsqueeze(1), Wgt2_real.unsqueeze(1)), dim=1), torch.cat((Wgt1_imag.unsqueeze(1), Wgt2_imag.unsqueeze(1)), dim=1)

def get_gtW_negative(Ht_real, Ht_imag, Hr_real, Hr_imag, eps = 1e-16):
    assert(Ht_real.size(1) == 2), 'currently, only #mic=2 is supported'

    Ht1_real = Ht_real[:, 0, :, :]
    Ht1_imag = Ht_imag[:, 0, :, :]
    Ht2_real = Ht_real[:, 1, :, :]
    Ht2_imag = Ht_imag[:, 1, :, :]

    Hr1_real = Hr_real[:, 0, :, :]
    Hr1_imag = Hr_imag[:, 0, :, :]
    Hr2_real = Hr_real[:, 1, :, :]
    Hr2_imag = Hr_imag[:, 1, :, :]

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

    return torch.cat((Wgt1_real.unsqueeze(1), Wgt2_real.unsqueeze(1)), dim=1), torch.cat((Wgt1_imag.unsqueeze(1), Wgt2_imag.unsqueeze(1)), dim=1)


def forward_common(input, net, Loss,
                   Loss2 = None, Eval=None, Eval2=None, loss2_type = '', eval_type = '', eval2_type='', 
                   use_ref_IR=False):

    tarH = input[0]
    tarH_real, tarH_imag = tarH[..., 0], tarH[..., 1]
    N, M, F, T, _ = tarH.size()

    if(use_ref_IR):
        refH = input[1]
        refH_real, refH_imag = refH[..., 0], refH[..., 1]

    West_real, West_imag = net(tarH_real, tarH_imag) # for now, refH is not used to predict W

    if(loss_type.find('positive') == -1):
        Wgt_real, Wgt_imag = get_gtW_negative(tarH_real, tarH_imag, refH_real, refH_imag)
    else:
        Wgt_real, Wgt_imag = get_gtW_positive(tarH_real, tarH_imag, refH_real, refH_imag)

    loss = -Loss(Wgt_real, Wgt_imag, West_real, West_imag)


    if(Loss2 is not None):
        assert(0), 'loss2 (regularization) is not available for now'
    else:
        loss2 = None

    if(Eval is not None):
        if(eval_type.find('positive') == -1):
            Wgt_real, Wgt_imag = get_gtW_negative(tarH_real, tarH_imag, refH_real, refH_imag)
        else:
            Wgt_real, Wgt_imag = get_gtW_positive(tarH_real, tarH_imag, refH_real, refH_imag)
        eval_metric = Eval(Wgt_real, Wgt_imag, West_real, West_imag)
    else:
        eval_metric = None

    if(Eval2 is not None):
        if(eval2_type.find('positive') == -1):
            Wgt_real, Wgt_imag = get_gtW_negative(tarH_real, tarH_imag, refH_real, refH_imag)
        else:
            Wgt_real, Wgt_imag = get_gtW_positive(tarH_real, tarH_imag, refH_real, refH_imag)
        eval2_metric = Eval2(Wgt_real, Wgt_imag, West_real, West_imag)
    else:
        eval2_metric = None

    return loss, loss2, eval_metric, eval2_metric