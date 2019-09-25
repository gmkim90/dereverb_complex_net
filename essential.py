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

def normalize(y):
    y = y/max(abs(y))
    return y

def forward_common(input, net, Loss, data_type, loss_type, eval_type, eval2_type, stride_product, mode='train', expnum=-1, fixed_src=False,
                   Eval=None, Eval2=None, fix_len_by_cl='input', count=0,  use_pos=False, save_activation=False, eps=1e-10, save_wav=False, istft=None):

    mixedSTFT, cleanSTFT, len_STFT_cl = input[0].cuda(), input[1].cuda(), input[2]
    if(mixedSTFT.dim() == 4): # for singleCH experiment
        mixedSTFT = mixedSTFT.unsqueeze(1)
    #pdb.set_trace()
    bsz, nCH, F, Tf, _ = mixedSTFT.size()
    #Hreal = None # default dummy variable

    if(stride_product > 0 and not Tf % stride_product == 1):
        nPad_time = stride_product*math.ceil(Tf/stride_product) - Tf + 1
        mixedSTFT = Func.pad(mixedSTFT, (0, 0, 0, nPad_time, 0, 0))  # (Fs,Fe,Ts,Te, real, imag)
        cleanSTFT = Func.pad(cleanSTFT, (0, 0, 0, nPad_time, 0, 0))

    clean_real, clean_imag = cleanSTFT[..., 0], cleanSTFT[..., 1]
    mixed_real, mixed_imag = mixedSTFT[..., 0], mixedSTFT[..., 1]

    #pdb.set_trace()

    ''' # suspicious memory leak
    net_inputs = [mixed_real, mixed_imag]
    if(use_pos):
        if(mode == 'train'):
            #net_inputs.append(input[6]) # source position
            net_inputs.append(input[3])  # source position
        elif(mode == 'generate'):
            #net_inputs.append(input[7])  # source position
            net_inputs.append(input[4])  # source position
    net_out_list = net(net_inputs)
    out_list, mask_list = net_out_list[0][0], net_out_list[0][1]
    '''

    #out_list, mask_list = net(mixed_real, mixed_imag)
    out_real, out_imag, mask_real, mask_imag = net(mixed_real, mixed_imag)

    Tmax_cl = clean_real.size(-1)
    if(fix_len_by_cl == 'eval'):
        #Tmax_rev = out_list[0].size(-1)
        Tmax_rev = out_real.size(-1)

        if(Tmax_cl < Tmax_rev):
            #for n in range(len(out_list)):
                #out_list[n] = out_list[n][:, :, :clean_real.size(-1)]
            out_real = out_real[:, :, :clean_real.size(-1)]
            out_imag = out_imag[:, :, :clean_real.size(-1)]
            mask_real = mask_real[:, :, :, :clean_real.size(-1)]
            mask_imag = mask_imag[:, :, :, :clean_real.size(-1)]
        elif(Tmax_cl > Tmax_rev):
            clean_real = clean_real[:, :, :Tmax_rev]
            clean_imag = clean_imag[:, :, :Tmax_rev]
            Tmax_cl = Tmax_rev

    '''
    for n in range(len(out_list)):
        for i, l in enumerate(len_STFT_cl): # zero padding to output audio
            out_list[n][i, :, min(l, Tmax_cl):] = 0
    '''
    for i, l in enumerate(len_STFT_cl):  # zero padding to output audio
        out_real[i, :, min(l, Tmax_cl):] = 0
        out_imag[i, :, min(l, Tmax_cl):] = 0
        mask_real[i, :, :, min(l, Tmax_cl):] = 0
        mask_imag[i, :, :, min(l, Tmax_cl):] = 0

    if(loss_type == 'sInvSDR_mag' or loss_type == 'SD_SDR_complex_ipd' or loss_type == 'SD_SDR_spec_RIconcat'):
        #loss = -Loss(clean_real, clean_imag, out_list)
        loss = -Loss(clean_real, clean_imag, out_real, out_imag) # loss = -SDR
    elif(loss_type == 'srcIndepSDR_Cproj_by_SShat' or loss_type == 'SI_SDR_spec_RIconcat' or loss_type == 'SI_SDR_complex_ipd'):
        loss = -Loss(clean_real, clean_imag, out_real, out_imag, len_STFT_cl)  # loss = -SDR
    elif(loss_type == 'srcIndepSDR_freqpower_by_enhanced'):
        #loss = Loss(out_real, out_imag, len_STFT_cl) # WRONG SIGN
        loss = -Loss(out_real, out_imag, len_STFT_cl) # loss = -SDR
    elif(loss_type == 'sInvSDR_time'):
        mixed_time, clean_time, len_time = input[3], input[4].cuda(), input[5]
        out_audio = istft(out_real.squeeze(), out_imag.squeeze(), mixed_time.size(-1))
        for i, l in enumerate(len_time):  # zero padding to output audio
            out_audio[i, l:] = 0
        loss = -Loss(clean_time, out_audio)
    elif(loss_type == 'srcIndepSDR_mag' or loss_type == 'srcIndepSDR_freqpower'
    or loss_type == 'srcIndepSDR_mag_diffperT' or loss_type == 'srcIndepSDR_freqpower_diffperT'
    or loss_type == 'srcIndepSDR_Cproj_by_WH'):
        #Wreal = mask_list[0]
        #Wimag = mask_list[1]
        Wreal = mask_real
        Wimag = mask_imag

        # ver 1. mixedSTFT/cleanSTFT
        #eps = 1e-10
        #eps = 1e-6 # given by function argument

        #Tset = out_list[0].size(-1)
        Tset = out_real.size(-1)
        clean_real_tmp = clean_real[:, :, :Tset].unsqueeze(1) # add mic dimension
        clean_imag_tmp = clean_imag[:, :, :Tset].unsqueeze(1)  # add mic dimension
        mixed_real = mixed_real[:, :, :, :Tset]
        mixed_imag = mixed_imag[:, :, :, :Tset]

        cleanSTFT_pow = clean_real_tmp*clean_real_tmp + clean_imag_tmp*clean_imag_tmp + eps

        Hreal = (mixed_real*clean_real_tmp + mixed_imag*clean_imag_tmp)/cleanSTFT_pow
        Himag = (-mixed_real*clean_imag_tmp + mixed_imag*clean_real_tmp)/cleanSTFT_pow


        #loss, Cmag = Loss(Wreal, Wimag, Hreal, Himag, len_STFT_cl)
        loss = -Loss(Wreal, Wimag, Hreal, Himag, len_STFT_cl) # loss = -SDR


    if(mode == 'generate' and save_activation): # generate spectroram
        specs_path = 'specs/' + str(expnum) + '/' + data_type + '_' + str(count) + '.mat'

        # # TEMPORARY ANALYSIS START
        # Tset = out_real.size(-1)
        # clean_real_tmp = clean_real[:, :, :Tset].unsqueeze(1) # add mic dimension
        # clean_imag_tmp = clean_imag[:, :, :Tset].unsqueeze(1)  # add mic dimension
        # mixed_real = mixed_real[:, :, :, :Tset]
        # mixed_imag = mixed_imag[:, :, :, :Tset]
        #
        # cleanSTFT_pow = clean_real_tmp*clean_real_tmp + clean_imag_tmp*clean_imag_tmp + eps
        #
        # Hreal = (mixed_real*clean_real_tmp + mixed_imag*clean_imag_tmp)/cleanSTFT_pow
        # Himag = (-mixed_real*clean_imag_tmp + mixed_imag*clean_real_tmp)/cleanSTFT_pow
        # Hpow = Hreal*Hreal + Himag*Himag
        #
        # Wpow = mask_real*mask_real + mask_imag*mask_imag
        #
        # Cr = (out_real * clean_real + out_imag * clean_imag) / cleanSTFT_pow
        # Ci = (-out_real * clean_imag + out_imag * clean_real) / cleanSTFT_pow
        # Cpow = Cr * Cr + Ci * Ci + eps
        #
        # sio.savemat(specs_path, {'Hpow':Hpow.data.cpu().numpy(),
        #                          'Wpow':Wpow.data.cpu().numpy(),
        #                         'Cpow': Cpow.data.cpu().numpy()})
        # # TEMPORARY ANALYSIS END

        if(not fixed_src or count == 0):
            sio.savemat(specs_path, {'mixed_real':mixed_real.data.cpu().numpy(), 'mixed_imag':mixed_imag.data.cpu().numpy(),
                                    'out_real': out_real.data.cpu().numpy(), 'out_imag':out_imag.data.cpu().numpy(),
                                    'clean_real': clean_real.data.cpu().numpy(), 'clean_imag':clean_imag.data.cpu().numpy(),
                                    'mask_real':mask_real.data.cpu().numpy(), 'mask_imag':mask_imag.data.cpu().numpy()})
        else:
            sio.savemat(specs_path, {'mixed_real':mixed_real.data.cpu().numpy(), 'mixed_imag':mixed_imag.data.cpu().numpy(),
                                    'out_real': out_real.data.cpu().numpy(), 'out_imag':out_imag.data.cpu().numpy(),
                                    'mask_real':mask_real.data.cpu().numpy(), 'mask_imag':mask_imag.data.cpu().numpy()})

    if(loss_type == eval_type):
        eval_metric = -loss
    elif(eval_type == 'srcIndepSDR_freqpower_by_enhanced'):
        eval_metric = Eval(out_real, out_imag, len_STFT_cl)
    elif(eval_type == 'srcIndepSDR_mag' or eval_type  == 'srcIndepSDR_freqpower'
    or eval_type == 'srcIndepSDR_mag_diffperT' or eval_type  == 'srcIndepSDR_freqpower_diffperT'
    or eval_type == 'srcIndepSDR_Cproj_by_WH'):
        #Wreal = mask_list[0]
        #Wimag = mask_list[1]
        Wreal = mask_real
        Wimag = mask_imag

        # ver 1. mixedSTFT/cleanSTFT
        #pdb.set_trace()
        #eps = 1e-10
        # eps = 1e-6 # given by function argument
        #Tset = out_list[0].size(-1)
        Tset = clean_real.size(-1)
        clean_real_tmp = clean_real.unsqueeze(1)  # add mic dimension
        clean_imag_tmp = clean_imag.unsqueeze(1)  # add mic dimension
        mixed_real = mixed_real[:, :, :, :Tset]
        mixed_imag = mixed_imag[:, :, :, :Tset]

        #pdb.set_trace()

        cleanSTFT_pow = clean_real_tmp * clean_real_tmp + clean_imag_tmp * clean_imag_tmp + eps

        Hreal = (mixed_real * clean_real_tmp + mixed_imag * clean_imag_tmp) / cleanSTFT_pow
        Himag = (-mixed_real * clean_imag_tmp + mixed_imag * clean_real_tmp) / cleanSTFT_pow

        #eval_metric, Cmag = Eval(Wreal, Wimag, Hreal, Himag, len_STFT_cl)
        eval_metric = Eval(Wreal, Wimag, Hreal, Himag, len_STFT_cl)
    elif(eval_type == 'srcIndepSDR_Cproj_by_SShat' or eval_type == 'SI_SDR_spec_RIconcat' or eval_type == 'SI_SDR_complex_ipd'):
        eval_metric = Eval(clean_real, clean_imag, out_real, out_imag, len_STFT_cl)
    else: # sInvSDR_mag
        #eval_metric = Eval(clean_real, clean_imag, out_list)
        eval_metric = Eval(clean_real, clean_imag, out_real, out_imag)


    if(loss_type == eval2_type):
        eval2_metric = -loss
    elif (eval2_type == ''):
        eval2_metric = None
    elif(eval2_type == 'srcIndepSDR_freqpower_by_enhanced'):
        eval2_metric = Eval2(out_real, out_imag, len_STFT_cl)
    elif(eval2_type == 'srcIndepSDR_mag' or eval2_type  == 'srcIndepSDR_freqpower'
    or eval2_type == 'srcIndepSDR_mag_diffperT' or eval2_type  == 'srcIndepSDR_freqpower_diffperT'
    or eval2_type == 'srcIndepSDR_Cproj_by_WH'):
        #Wreal = mask_list[0]
        #Wimag = mask_list[1]
        Wreal = mask_real
        Wimag = mask_imag

        # ver 1. mixedSTFT/cleanSTFT
        #pdb.set_trace()
        #eps = 1e-10
        #eps = 1e-6 # given by function argument

        #Tset = out_list[0].size(-1)
        Tset = clean_real.size(-1)
        clean_real_tmp = clean_real.unsqueeze(1)  # add mic dimension
        clean_imag_tmp = clean_imag.unsqueeze(1)  # add mic dimension
        mixed_real = mixed_real[:, :, :, :Tset]
        mixed_imag = mixed_imag[:, :, :, :Tset]

        #pdb.set_trace()

        cleanSTFT_pow = clean_real_tmp * clean_real_tmp + clean_imag_tmp * clean_imag_tmp + eps

        Hreal = (mixed_real * clean_real_tmp + mixed_imag * clean_imag_tmp) / cleanSTFT_pow
        Himag = (-mixed_real * clean_imag_tmp + mixed_imag * clean_real_tmp) / cleanSTFT_pow

        #eval_metric, Cmag = Eval(Wreal, Wimag, Hreal, Himag, len_STFT_cl)
        eval2_metric = Eval2(Wreal, Wimag, Hreal, Himag, len_STFT_cl)
    elif (eval2_type == 'srcIndepSDR_Cproj_by_SShat' or eval2_type == 'SI_SDR_spec_RIconcat' or eval2_type == 'SI_SDR_complex_ipd'):
        eval2_metric = Eval2(clean_real, clean_imag, out_real, out_imag, len_STFT_cl)
    else: # sInvSDR_mag
        #eval_metric = Eval(clean_real, clean_imag, out_list)
        eval2_metric = Eval2(clean_real, clean_imag, out_real, out_imag)

    #if(loss_type == 'srcIndepSDR_mag' or loss_type == 'srcIndepSDR_freqpower'):
        #return loss, eval_metric, eval2_metric, Cmag


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

    return loss, eval_metric, eval2_metric