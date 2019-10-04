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
                   Loss2 = None, Eval=None, Eval2=None, eval2_type='', fix_len_by_cl='input', count=0, save_activation=False, save_wav=False, istft=None,
                   use_ref_IR=False, use_neighbor_IR=False):

    mixedSTFT, cleanSTFT, len_STFT_cl = input[0].cuda(), input[1].cuda(), input[2]
    if(mixedSTFT.dim() == 4): # for singleCH experiment
        mixedSTFT = mixedSTFT.unsqueeze(1)
    bsz, nCH, F, Tf, _ = mixedSTFT.size()

    if(use_ref_IR):
        refmicSTFT = input[3].cuda()
    elif(use_neighbor_IR):
        nbmicSTFT = input[3].cuda()
        nbmicSTFT = nbmicSTFT.view(-1, nCH, F, nbmicSTFT.size(3), 2)


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
            Wgt_real, Wgt_imag = get_gtW(mixed_real, mixed_imag, refmic_real, refmic_imag, clean_real, clean_imag)
            loss = -Loss(Wgt_real, Wgt_imag, mask_real, mask_imag, len_STFT_cl)
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

    if(Loss2 is not None):
        loss2 = -Loss2(refmic_real, refmic_imag, mask_real, mask_imag, len_STFT_cl)
    else:
        loss2 = None

    if(Eval is not None):
        if(Eval == Loss):
            eval_metric = -loss
        else:
            eval_metric = Eval(clean_real, clean_imag, out_real, out_imag, len_STFT_cl)

    if(Eval2 is not None):
        if(eval2_type.find('Wdiff') == -1):
            eval2_metric = Eval2(clean_real, clean_imag, out_real, out_imag, len_STFT_cl)
        else:
            Wgt_real, Wgt_imag = get_gtW(mixed_real, mixed_imag, refmic_real, refmic_imag, clean_real, clean_imag)
            eval2_metric = Eval2(Wgt_real, Wgt_imag, mask_real, mask_imag, len_STFT_cl)
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