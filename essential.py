import os

import torch
import torch.nn.functional as Func
#import librosa
#import torchaudio
import soundfile as sf
import math
import scipy.io as sio

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


def get_gtW_positive_src(Xt_real, Xt_imag, Xr_real, Xr_imag, S_real, S_imag, eps = 1e-16):
    assert(Xt_real.size(1) == 2), 'currently, only #mic=2 is supported'

    Xt1_real = Xt_real[:, 0, :, :]
    Xt1_imag = Xt_imag[:, 0, :, :]
    Xt2_real = Xt_real[:, 1, :, :]
    Xt2_imag = Xt_imag[:, 1, :, :]

    Xr1_real = Xr_real[:, 0, :, :]
    Xr1_imag = Xr_imag[:, 0, :, :]
    Xr2_real = Xr_real[:, 1, :, :]
    Xr2_imag = Xr_imag[:, 1, :, :]

    # determinant
    det_real = (Xt1_real * Xr2_real - Xt1_imag * Xr2_imag) - (Xt2_real * Xr1_real - Xt2_imag * Xr1_imag)  # NxFxT
    det_imag = (Xt1_real * Xr2_imag + Xt1_imag * Xr2_real) - (Xt2_real * Xr1_imag + Xt2_imag * Xr1_real)  # NxFxT

    det_power = det_real * det_real + det_imag * det_imag

    # S/det
    S_det_real = (S_real * det_real + S_imag * det_imag) / (det_power+eps)
    S_det_imag = (S_imag * det_real - S_real * det_imag) / (det_power+eps)

    # multiply Xref (=Wgt)
    Wgt1_real = S_det_real * (Xr2_real-Xt2_real) - S_det_imag * (Xr2_imag-Xt2_imag)
    Wgt1_imag = S_det_real * (Xr2_imag-Xt2_imag) + S_det_imag * (Xr2_real-Xt2_real)
    Wgt2_real = S_det_real * (-Xr1_real+Xt1_real) - S_det_imag * (-Xr1_imag+Xt1_imag)
    Wgt2_imag = S_det_real * (-Xr1_imag+Xt1_imag) + S_det_imag * (-Xr1_real+Xt1_real)

    #return Wgt_real, Wgt_imag
    return torch.cat((Wgt1_real.unsqueeze(1), Wgt2_real.unsqueeze(1)), dim=1), torch.cat((Wgt1_imag.unsqueeze(1), Wgt2_imag.unsqueeze(1)), dim=1)

def get_gtW_negative_src(Xt_real, Xt_imag, Xr_real, Xr_imag, S_real, S_imag, eps = 1e-16):
    assert(Xt_real.size(1) == 2), 'currently, only #mic=2 is supported'

    Xt1_real = Xt_real[:, 0, :, :]
    Xt1_imag = Xt_imag[:, 0, :, :]
    Xt2_real = Xt_real[:, 1, :, :]
    Xt2_imag = Xt_imag[:, 1, :, :]

    Xr1_real = Xr_real[:, 0, :, :]
    Xr1_imag = Xr_imag[:, 0, :, :]
    Xr2_real = Xr_real[:, 1, :, :]
    Xr2_imag = Xr_imag[:, 1, :, :]

    # determinant
    det_real = (Xt1_real * Xr2_real - Xt1_imag * Xr2_imag) - (Xt2_real * Xr1_real - Xt2_imag * Xr1_imag)  # NxFxT
    det_imag = (Xt1_real * Xr2_imag + Xt1_imag * Xr2_real) - (Xt2_real * Xr1_imag + Xt2_imag * Xr1_real)  # NxFxT

    det_power = det_real * det_real + det_imag * det_imag

    # S/det
    S_det_real = (S_real * det_real + S_imag * det_imag) / (det_power+eps)
    S_det_imag = (S_imag * det_real - S_real * det_imag) / (det_power+eps)

    # multiply Xref (=Wgt)
    Wgt1_real = S_det_real * Xr2_real - S_det_imag * Xr2_imag
    Wgt1_imag = S_det_real * Xr2_imag + S_det_imag * Xr2_real
    Wgt2_real = S_det_real * (-Xr1_real) - S_det_imag * (-Xr1_imag)
    Wgt2_imag = S_det_real * (-Xr1_imag) + S_det_imag * (-Xr1_real)

    #return Wgt_real, Wgt_imag
    return torch.cat((Wgt1_real.unsqueeze(1), Wgt2_real.unsqueeze(1)), dim=1), torch.cat((Wgt1_imag.unsqueeze(1), Wgt2_imag.unsqueeze(1)), dim=1)

def forward_common(input, net, Loss,
                   Loss2 = None, Eval=None, Eval2=None, loss_type = '', loss2_type = '', eval_type = '', eval2_type='',
                   use_ref_IR=False, save_activation=False, savename='', freq_center_idx=-1, freq_context_left_right_idx=0,
                   match_domain='realimag'):

    tarH = input[0]
    tarH_real, tarH_imag = tarH[..., 0], tarH[..., 1]
    N, M, F, T, _ = tarH.size()

    if(use_ref_IR):
        refH = input[1]
        refH_real, refH_imag = refH[..., 0], refH[..., 1]

    if (freq_center_idx >= 0 and freq_context_left_right_idx > 0):
        tarH_real = tarH_real[:, :, freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx+1, :]
        tarH_imag = tarH_imag[:, :,freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx + 1, :]
        refH_real = refH_real[:, :,freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx + 1, :]
        refH_imag = refH_imag[:, :,freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx + 1, :]

    West_real, West_imag = net(tarH_real, tarH_imag) # for now, refH is not used to predict W

    if(loss_type.find('Wdiff') >= 0):
        if(loss_type.find('positive') == -1):
            Wgt_real, Wgt_imag = get_gtW_negative(tarH_real, tarH_imag, refH_real, refH_imag)
        else:
            Wgt_real, Wgt_imag = get_gtW_positive(tarH_real, tarH_imag, refH_real, refH_imag)
        #if(freq_center_idx >= 0 and freq_context_left_right_idx > 0):
#            Wgt_real = Wgt_real[:, :, freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx+1, :]
#            Wgt_imag = Wgt_imag[:, :, freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx+1, :]

        loss = -Loss(Wgt_real, Wgt_imag, West_real, West_imag)
    elif(loss_type.find('WH_sum') >= 0):
        if (loss_type.find('positive') == -1):
            target_real = 0
            target_imag = 0
        else:
            target_real = 1
            target_imag = 0

        if(loss_type.find('tar') >= 0):
            H_real = tarH_real
            H_imag = tarH_imag
        elif(loss_type.find('ref')>= 0):
            H_real = refH_real
            H_imag = refH_imag
        loss = -Loss(H_real, H_imag, West_real, West_imag, target_real, target_imag, match_domain = match_domain, Tlist=None)
    else:
        loss = None

    if(Loss2 is not None):
        if (loss2_type.find('Wdiff') >= 0):
            if (loss2_type.find('positive') == -1):
                Wgt_real, Wgt_imag = get_gtW_negative(tarH_real, tarH_imag, refH_real, refH_imag)
            else:
                Wgt_real, Wgt_imag = get_gtW_positive(tarH_real, tarH_imag, refH_real, refH_imag)
            if (freq_center_idx >= 0 and freq_context_left_right_idx > 0):
                Wgt_real = Wgt_real[:, :, freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx + 1,:]
                Wgt_imag = Wgt_imag[:, :,freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx + 1,:]

            loss2 = -Loss2(Wgt_real, Wgt_imag, West_real, West_imag)
        elif (loss2_type.find('WH_sum') >= 0):
            if (loss2_type.find('positive') == -1):
                target_real = 0
                target_imag = 0
            else:
                target_real = 1
                target_imag = 0

            if (loss2_type.find('tar') >= 0):
                H_real = tarH_real
                H_imag = tarH_imag
            elif (loss2_type.find('ref') >= 0):
                H_real = refH_real
                H_imag = refH_imag
            loss2 = -Loss2(H_real, H_imag, West_real, West_imag, target_real, target_imag, match_domain = match_domain,  Tlist=None)
        else:
            loss2 = None
    else:
        loss2 = None

    if(Eval is not None):
        if (eval_type.find('Wdiff') >= 0):
            if (eval_type.find('positive') == -1):
                Wgt_real, Wgt_imag = get_gtW_negative(tarH_real, tarH_imag, refH_real, refH_imag)
            else:
                Wgt_real, Wgt_imag = get_gtW_positive(tarH_real, tarH_imag, refH_real, refH_imag)
            if (freq_center_idx >= 0 and freq_context_left_right_idx > 0):
                Wgt_real = Wgt_real[:, :, freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx + 1,:]
                Wgt_imag = Wgt_imag[:, :,freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx + 1,:]

            eval_metric = Eval(Wgt_real, Wgt_imag, West_real, West_imag)
        elif (eval_type.find('WH_sum') >= 0):
            if (eval_type.find('positive') == -1):
                target_real = 0
                target_imag = 0
            else:
                target_real = 1
                target_imag = 0

            if (eval_type.find('tar') >= 0):
                H_real = tarH_real
                H_imag = tarH_imag
            elif (eval_type.find('ref') >= 0):
                H_real = refH_real
                H_imag = refH_imag
            eval_metric = Eval(H_real, H_imag, West_real, West_imag, target_real, target_imag, match_domain = match_domain, Tlist=None)
        else:
            eval_metric = None
    else:
        eval_metric = None

    if(Eval2 is not None):
        if (eval2_type.find('Wdiff') >= 0):
            if (eval2_type.find('positive') == -1):
                Wgt_real, Wgt_imag = get_gtW_negative(tarH_real, tarH_imag, refH_real, refH_imag)
            else:
                Wgt_real, Wgt_imag = get_gtW_positive(tarH_real, tarH_imag, refH_real, refH_imag)
            if (freq_center_idx >= 0 and freq_context_left_right_idx > 0):
                Wgt_real = Wgt_real[:, :, freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx + 1,:]
                Wgt_imag = Wgt_imag[:, :,freq_center_idx - freq_context_left_right_idx:freq_center_idx + freq_context_left_right_idx + 1,:]

            eval2_metric = Eval2(Wgt_real, Wgt_imag, West_real, West_imag)
        elif (eval2_type.find('WH_sum') >= 0):
            if (eval2_type.find('positive') == -1):
                target_real = 0
                target_imag = 0
            else:
                target_real = 1
                target_imag = 0

            if (eval2_type.find('tar') >= 0):
                H_real = tarH_real
                H_imag = tarH_imag
            elif (eval2_type.find('ref') >= 0):
                H_real = refH_real
                H_imag = refH_imag
            eval2_metric = Eval2(H_real, H_imag, West_real, West_imag, target_real, target_imag, match_domain = match_domain, Tlist=None)
        else:
            eval2_metric = None
    else:
        eval2_metric = None

    if(save_activation): # separate activation & metric to save memory
        save_dict = {}
        save_dict['tarH'] = tarH.data.cpu().numpy()
        save_dict['West_real'] = West_real.data.cpu().numpy()
        save_dict['West_imag'] = West_imag.data.cpu().numpy()
        if(use_ref_IR):
            save_dict['refH'] = refH.data.cpu().numpy()
            save_dict['Wgt_real'] = Wgt_real.data.cpu().numpy()
            save_dict['Wgt_imag'] = Wgt_imag.data.cpu().numpy()

        # source-free experiment
        sio.savemat(savename, save_dict)
    return loss, loss2, eval_metric, eval2_metric

def forward_common_src(input, net, Loss, stride_product, out_type,
                   Loss2 = None, Eval=None, Eval2=None, loss_type = '', loss2_type = '', eval_type = '', eval2_type='',
                   use_ref_IR=False, save_activation=False, savename='', match_domain='realimag'):
    mic_STFT, clean_STFT, Tlist = input[0], input[1], input[2]
    #print(Tlist)

    if(use_ref_IR):
        refmic_STFT = input[3]

    T = clean_STFT.size(2)

    if (stride_product > 0 and not T % stride_product == 1):
        nPad_frame = stride_product*math.ceil(T/stride_product) - T + 1
        mic_STFT = Func.pad(mic_STFT, (0, 0, 0, nPad_frame, 0, 0))  # (Fs,Fe,Ts,Te, real, imag)
        clean_STFT = Func.pad(clean_STFT, (0, 0, 0, nPad_frame, 0, 0))
        if(use_ref_IR):
            refmic_STFT = Func.pad(refmic_STFT, (0, 0, 0, nPad_frame, 0, 0))

    mic_real, mic_imag, clean_real, clean_imag = mic_STFT[..., 0], mic_STFT[..., 1], clean_STFT[..., 0], clean_STFT[..., 1]
    if(use_ref_IR):
        refmic_real, refmic_imag = refmic_STFT[..., 0], refmic_STFT[..., 1]

    # Forward
    if(not save_activation):
        out_real, out_imag = net(mic_real, mic_imag)
    else:
        if(net.input_type == 'complex_ratio'):
            out_real, out_imag, IMR_real, IMR_imag = net(mic_real, mic_imag, return_IMR=True)
    if (out_type == 'W'):
        enh_real = torch.sum(mic_real * out_real - mic_imag * out_imag, dim=1)  # NxFxT
        enh_imag = torch.sum(mic_real * out_imag + mic_imag * out_real, dim=1)  # NxFxT
        mask_real, mask_imag = out_real, out_imag
    elif(out_type == 'S'):
        enh_real = out_real
        enh_imag = out_imag
        mask_real, mask_imag = None, None

    maxT = enh_real.size(-1)
    for i, l in enumerate(Tlist):  # zero padding to output audio
        enh_real[i, :, min(l, maxT):] = 0
        enh_imag[i, :, min(l, maxT):] = 0
        if (out_type == 'W'):
            mask_real[i, :, :, min(l, maxT):] = 0
            mask_imag[i, :, :, min(l, maxT):] = 0

    # Metrics
    if(loss_type.find('Wdiff') == -1): # loss not on W
        loss = -Loss(enh_real, enh_imag, clean_real, clean_imag, Tlist, match_domain=match_domain) # note that we only allow positive target loss
    else:
        Wgt_real, Wgt_imag = get_gtW_negative_src(mic_real, mic_imag, refmic_real, refmic_imag, clean_real, clean_imag)
        loss = -Loss(Wgt_real, Wgt_imag, mask_real, mask_imag)

    if(Loss2 is not None):
        if(loss2_type.find('Cdistortion') >= 0):
            enh_real_ref = torch.sum(refmic_real * mask_real - refmic_imag * mask_imag, dim=1)  # NxFxT
            enh_imag_ref = torch.sum(refmic_real * mask_imag + refmic_imag * mask_real, dim=1)  # NxFxT
            loss2 = -Loss2(enh_real_ref, enh_imag_ref, clean_real, clean_imag, Tlist, match_domain=match_domain)  # note that we only allow positive target loss

        else:
            if (loss2_type.find('positive') >= 0):
                target_real = clean_real
                target_imag = clean_imag
            elif (loss2_type.find('negative') >= 0):
                target_real = 0
                target_imag = 0

            if (loss2_type.find('tar') >= 0):
                in_real = mic_real
                in_imag = mic_imag
            elif (loss2_type.find('ref') >= 0):
                in_real = refmic_real
                in_imag = refmic_imag

            loss2 = -Loss2(in_real, in_imag, out_real, out_imag, target_real, target_imag, Tlist, match_domain=match_domain)

    else:
        loss2 = None

    if(Eval is not None): # SDR(|S|)
        eval_metric = Eval(clean_real, clean_imag, enh_real, enh_imag, Tlist)
    else:
        eval_metric = None

    if(Eval2 is not None): # SDR(|C|) or SDR(W)
        if(eval2_type == 'SDR_C_mag'):
            eval2_metric = Eval2(clean_real, clean_imag, enh_real, enh_imag, Tlist)
        else:
            if(loss_type.find('Wdiff') == 0): # if loss calculates Wgt, use them
                if (eval2_type.find('negative') >= 0):
                    Wgt_real, Wgt_imag = get_gtW_negative_src(mic_real, mic_imag, refmic_real, refmic_imag, clean_real, clean_imag)
                elif(eval2_type.find('positive') >= 0):
                    Wgt_real, Wgt_imag = get_gtW_positive_src(mic_real, mic_imag, refmic_real, refmic_imag, clean_real, clean_imag)

            eval2_metric = Eval2(Wgt_real, Wgt_imag, mask_real, mask_imag)
    else:
        eval2_metric = None

    if(save_activation): # separate activation & metric to save memory
        save_dict = {}
        save_dict['mic_STFT'] = mic_STFT.data.cpu().numpy()
        save_dict['enh_real'] = enh_real.data.cpu().numpy()
        save_dict['enh_imag'] = enh_imag.data.cpu().numpy()
        save_dict['clean_real'] = clean_real.data.cpu().numpy()
        save_dict['clean_imag'] = clean_imag.data.cpu().numpy()

        if(out_type == 'W'):
            save_dict['mask_real'] = mask_real.data.cpu().numpy()
            save_dict['mask_imag'] = mask_imag.data.cpu().numpy()

        if(use_ref_IR):
            save_dict['refmic_STFT'] = refmic_STFT.data.cpu().numpy()

        if(net.input_type == 'complex_ratio'):
            save_dict['IMR_real'] = IMR_real.data.cpu().numpy()
            save_dict['IMR_imag'] = IMR_imag.data.cpu().numpy()

        sio.savemat(savename, save_dict)

    return loss, loss2, eval_metric, eval2_metric