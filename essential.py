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


def TVN(xr, xi, Tlist, eps=1e-16): # Temporal variance normalization
    x = torch.sqrt(xr * xr + xi * xi + eps)
    Tlist_float = Tlist.float().unsqueeze(1).cuda()
    n = x.ndim
    if (n == 4):  # NxMxFxT
        x_mic_mean = x.mean(1)
        t_mean = x_mic_mean.sum(2) / Tlist_float  # NxF
        t_sqmean = x_mic_mean.pow(2).sum(2) / Tlist_float  # NxF
        t_var = t_sqmean - t_mean * t_mean  # NxF
        t_std = t_var.sqrt().unsqueeze(1).unsqueeze(3)

    elif (n == 3):  # NxFxT
        t_mean = x.sum(2) / Tlist_float  # NxF
        t_sqmean = x.pow(2).sum(2) / Tlist_float  # NxF
        t_var = t_sqmean - t_mean * t_mean  # NxF
        t_std = t_var.sqrt().unsqueeze(2)

    yr = xr / t_std
    yi = xi / t_std
    return yr, yi

def get_gtW_positive(Xt_real, Xt_imag, Xr_real, Xr_imag, S_real, S_imag, eps = 1e-16):
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

def get_gtW(Xt_real, Xt_imag, Xr_real, Xr_imag, S_real, S_imag, eps = 1e-16):
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

def get_cmvn_per_freq(W):
    # W: NxMxFxT
    Wmean = W.mean(3).mean(0).unsqueeze(0).unsqueeze(3) # 1xMxFx1
    Wstd = W.mean(3).std(0).unsqueeze(0).unsqueeze(3) # 1xMxFx1

    return Wmean, Wstd

def normalize(y):
    y = y/max(abs(y))
    return y

def neighbor_sensitivitiy(loss, loss_neighbor, nNeighbor):
    # input
    # loss: Nx1
    # loss_neighbor: sum_n(nNeighbor)x1
    # nNeighbor

    # output
    # neighbor_sensitivity:  (loss - neighbor_average_loss)^2 --> dim = Nx1

    # IMPORTANT NOTE: to exclude neighborhood sample from backward path, use detach() on loss_neighbor
    loss_neighbor = loss_neighbor.detach()

    neighbor_sensitivity = 0
    return neighbor_sensitivity

def forward_common(input, net, Loss, data_type, loss_type, stride_product, mode='train', expnum=-1,
                   Loss2 = None, Eval=None, Eval2=None, loss2_type = '', eval_type = '', eval2_type='', fix_len_by_cl='input', count=0, save_activation=False, save_wav=False, istft=None,
                   use_ref_IR=False, use_neighbor_IR=False, use_TVN_x=False, use_TVN_s=False):

    mixedSTFT, cleanSTFT, len_STFT_cl = input[0].cuda(), input[1].cuda(), input[2]
    if(mixedSTFT.dim() == 4): # for singleCH experiment
        mixedSTFT = mixedSTFT.unsqueeze(1)
    bsz, nCH, F, Tf, _ = mixedSTFT.size()

    if(use_ref_IR):
        refmicSTFT = input[3].cuda()
    elif(use_neighbor_IR):
        nbmicSTFT = input[3].cuda()
        nbmicSTFT = nbmicSTFT.view(-1, nCH, F, nbmicSTFT.size(3), 2)

    #mixedSTFT = mixedSTFT[:, :, :, 3:5, :] # select t=4, 5
    #cleanSTFT = cleanSTFT[:, :, 3:5, :] # select t=4, 5
    #refmicSTFT = refmicSTFT[:, :, :, 3:5, :] # select t=4, 5



    if(stride_product > 0 and not Tf % stride_product == 1):
        nPad_time = stride_product*math.ceil(Tf/stride_product) - Tf + 1
        mixedSTFT = Func.pad(mixedSTFT, (0, 0, 0, nPad_time, 0, 0))  # (Fs,Fe,Ts,Te, real, imag)
        cleanSTFT = Func.pad(cleanSTFT, (0, 0, 0, nPad_time, 0, 0))
        if(use_ref_IR):
            refmicSTFT = Func.pad(refmicSTFT, (0, 0, 0, nPad_time, 0, 0))
        if(use_neighbor_IR):
            nbmicSTFT = Func.pad(nbmicSTFT, (0, 0, 0, nPad_time, 0, 0))

    if(use_ref_IR):
        refmic_real, refmic_imag = refmicSTFT[..., 0], refmicSTFT[..., 1]

    if(use_neighbor_IR):
        nbmic_real, nbmic_imag = nbmicSTFT[..., 0], nbmicSTFT[..., 1]

    clean_real, clean_imag = cleanSTFT[..., 0], cleanSTFT[..., 1]
    mixed_real, mixed_imag = mixedSTFT[..., 0], mixedSTFT[..., 1]

    if(use_TVN_x):
        mixed_real, mixed_imag = TVN(mixed_real, mixed_imag, len_STFT_cl)
        if(use_ref_IR):
            refmic_real, refmic_imag = TVN(refmic_real, refmic_imag, len_STFT_cl)

    if(use_TVN_s):
        clean_real, clean_imag = TVN(clean_real, clean_imag, len_STFT_cl)

    out_real, out_imag, mask_real, mask_imag = net(mixed_real, mixed_imag)
    if(use_neighbor_IR):
        out_nb_real, out_nb_imag, mask_nb_real, mask_nb_imag = net(nbmic_real, nbmic_imag)

    Tmax_cl = clean_real.size(-1)
    Tmax_rev = out_real.size(-1)
    minT = min(Tmax_cl, Tmax_rev)
    if (use_ref_IR):
        minT = min(minT, refmic_real.size(-1))

    if(fix_len_by_cl == 'eval'): # note that mic length = output length in this mode
        out_real = out_real[:, :, :minT]
        out_imag = out_imag[:, :, :minT]
        mask_real = mask_real[:, :, :, :minT]
        mask_imag = mask_imag[:, :, :, :minT]
        clean_real = clean_real[:, :, :minT]
        clean_imag = clean_imag[:, :, :minT]
        if(use_ref_IR):
            refmic_real = refmic_real[:, :, :, :minT]
            refmic_imag = refmic_imag[:, :, :, :minT]
        if(use_neighbor_IR):
            out_nb_real = out_nb_real[:, :, :minT]
            out_nb_imag = out_nb_imag[:, :, :minT]

    for i, l in enumerate(len_STFT_cl):  # zero padding to output audio
        out_real[i, :, min(l, minT):] = 0
        out_imag[i, :, min(l, minT):] = 0
        mask_real[i, :, :, min(l, minT):] = 0
        mask_imag[i, :, :, min(l, minT):] = 0

    if(not loss_type == 'sInvSDR_time'):
        if(loss_type.find('Wdiff') == -1):
            loss = -Loss(clean_real, clean_imag, out_real, out_imag, len_STFT_cl) # loss = -SDR
        else:
            if(use_ref_IR):
                if(loss_type.find('positive') == -1):
                    Wgt_real, Wgt_imag = get_gtW(mixed_real, mixed_imag, refmic_real, refmic_imag, clean_real, clean_imag)
                else:
                    Wgt_real, Wgt_imag = get_gtW_positive(mixed_real, mixed_imag, refmic_real, refmic_imag, clean_real, clean_imag)

                # do cmvn
                #Wgt_real_mean, Wgt_real_std = get_cmvn_per_freq(Wgt_real)
                #Wgt_imag_mean, Wgt_imag_std = get_cmvn_per_freq(Wgt_imag)
                #mask_real = mask_real*Wgt_real_std + Wgt_real_mean
                #mask_imag = mask_imag*Wgt_imag_std + Wgt_imag_mean

                loss = -Loss(Wgt_real, Wgt_imag, mask_real, mask_imag, len_STFT_cl)
                #sio.savemat('Wgt.mat', {'Wgt_real': Wgt_real.data.cpu().numpy(), 'Wgt_imag': Wgt_imag.data.cpu().numpy()})
                #sio.savemat('West.mat', {'West_real': mask_real.data.cpu().numpy(), 'mask_imag': mask_imag.data.cpu().numpy()})
                #pdb.set_trace()
            else:
                loss = torch.zeros(8,1).cuda()
    else:
        mixed_time, clean_time, len_time = input[3], input[4].cuda(), input[5]
        out_audio = istft(out_real.squeeze(), out_imag.squeeze(), mixed_time.size(-1))
        for i, l in enumerate(len_time):  # zero padding to output audio
            out_audio[i, l:] = 0
        loss = -Loss(clean_time, out_audio)

    if(use_neighbor_IR):
        # TODO: define clean_nb_real, clean_nb_imag, len_STFT_nb_cl
        loss_neighbor = -Loss(clean_nb_real, clean_nb_imag, out_nb_real, out_nb_imag, len_STFT_nb_cl)

        # TODO: save nNeighbor in se_dataset
        # TODO: make neighbor_sensitivity function
        # TODO: return loss_sensitivity loss & include at the training
        loss_sensitivity = neighbor_sensitivitiy(loss, loss_neighbor, nNeighbor)

    #pdb.set_trace()
    if(Loss2 is not None):
        if(use_ref_IR):
            if(loss2_type == 'reference_position_demixing'):
                loss2 = -Loss2(refmic_real, refmic_imag, mask_real, mask_imag, len_STFT_cl)
            elif(loss2_type == 'refIR_demix_positive'):
                loss2 = -Loss2(refmic_real, refmic_imag, mask_real, mask_imag, clean_real, clean_imag, len_STFT_cl)
        else:
            #loss2 = -Loss2(clean_real, clean_imag, out_real, out_imag, len_STFT_cl)
            loss2 = None
    else:
        loss2 = None

    if(Eval is not None):
        if(eval_type.find('Wdiff') == -1):
            eval_metric = Eval(clean_real, clean_imag, out_real, out_imag, len_STFT_cl)
        else:
            if(use_ref_IR):
                if (loss2_type.find('positive') == -1): # depends on loss2 type !!
                    Wgt_real, Wgt_imag = get_gtW(mixed_real, mixed_imag, refmic_real, refmic_imag, clean_real, clean_imag)
                else:
                    Wgt_real, Wgt_imag = get_gtW_positive(mixed_real, mixed_imag, refmic_real, refmic_imag, clean_real, clean_imag)
                eval_metric = Eval(Wgt_real, Wgt_imag, mask_real, mask_imag, len_STFT_cl)
            else:
                eval_metric = None

    if(Eval2 is not None):
        if(eval2_type.find('Wdiff') == -1):
            eval2_metric = Eval2(clean_real, clean_imag, out_real, out_imag, len_STFT_cl)
        else:
            if(use_ref_IR):
                if (loss2_type.find('positive') == -1): # depends on loss2 type !!
                    Wgt_real, Wgt_imag = get_gtW(mixed_real, mixed_imag, refmic_real, refmic_imag, clean_real, clean_imag)
                else:
                    Wgt_real, Wgt_imag = get_gtW_positive(mixed_real, mixed_imag, refmic_real, refmic_imag, clean_real, clean_imag)
                eval2_metric = Eval2(Wgt_real, Wgt_imag, mask_real, mask_imag, len_STFT_cl)
            else:
                eval2_metric = None
    else:
        eval2_metric = None


    if(mode == 'generate' and save_activation): # generate spectroram
        specs_path = 'specs/' + str(expnum) + '/' + data_type + '_' + str(count) + '.mat'

        if(use_ref_IR): # calculate ground-truth W
            # given
            # clean: NxFxT
            # targetIR mic(=mixed): NxMxFxT
            # referIR mic(=refmic): NxMxFxT
            Wgt_real, Wgt_imag = get_gtW(mixed_real, mixed_imag, refmic_real, refmic_imag, clean_real, clean_imag)

            sio.savemat(specs_path, {'mixed_real':mixed_real.data.cpu().numpy(), 'mixed_imag':mixed_imag.data.cpu().numpy(),
                                    'out_real': out_real.data.cpu().numpy(), 'out_imag':out_imag.data.cpu().numpy(),
                                    'clean_real': clean_real.data.cpu().numpy(), 'clean_imag':clean_imag.data.cpu().numpy(),
                                    'mask_real':mask_real.data.cpu().numpy(), 'mask_imag':mask_imag.data.cpu().numpy(),
                                     'Wgt_real':Wgt_real.data.cpu().numpy(),'Wgt2_imag':Wgt_imag.data.cpu().numpy(),
                                     })
        else:
            sio.savemat(specs_path, {'mixed_real':mixed_real.data.cpu().numpy(), 'mixed_imag':mixed_imag.data.cpu().numpy(),
                                    'out_real': out_real.data.cpu().numpy(), 'out_imag':out_imag.data.cpu().numpy(),
                                    'clean_real': clean_real.data.cpu().numpy(), 'clean_imag':clean_imag.data.cpu().numpy(),
                                    'mask_real':mask_real.data.cpu().numpy(), 'mask_imag':mask_imag.data.cpu().numpy()})

    if(save_wav and not data_type == 'train'):
        if(not loss_type == 'sInvSDR_time'):
            mixed_time, clean_time, len_time = input[3], input[4], input[5]
            out_audio = istft(out_real.squeeze(), out_imag.squeeze(), mixed_time.size(-1))
            for i, l in enumerate(len_time):  # zero padding to output audio
                out_audio[i, l:] = 0

        # write wav
        T0 = len_time[0].item()
        sf.write('wavs/' + str(expnum) + '/mixed_' + data_type + '.wav', mixed_time[0][0][:T0].data.cpu().numpy(), 16000)
        sf.write('wavs/' + str(expnum) + '/clean_' + data_type + '.wav',clean_time[0][:T0].data.cpu().numpy(), 16000)
        sf.write('wavs/' + str(expnum) + '/out_' + data_type + '.wav', out_audio[0][:T0].data.cpu().numpy(),16000)
        sf.write('wavs/' + str(expnum) + '/mixed_' + data_type + '_norm.wav', normalize(mixed_time[0][0][:T0]).data.cpu().numpy(), 16000)
        sf.write('wavs/' + str(expnum) + '/clean_' + data_type + '_norm.wav', normalize(clean_time[0][:T0]).data.cpu().numpy(), 16000)
        sf.write('wavs/' + str(expnum) + '/out_' + data_type + '_norm.wav', normalize(out_audio[0][:T0]).data.cpu().numpy(),16000)

    return loss, loss2, eval_metric, eval2_metric