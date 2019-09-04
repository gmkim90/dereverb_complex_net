import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa
import math

import utils

import pdb

#n_fft = args.n_fft
#hop_length = args.hop_length
#window = torch.hann_window(n_fft).cuda()
#stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
#istft = ISTFT(n_fft, hop_length, window='hanning').cuda()

def forward_common(stft, istft, input, net, Loss, data_type, loss_type, stride_product, write_wav=False, expnum=-1, all_wav=False, Eval=None, eval_type=None):
    #train_mixed, train_clean, seq_len = map(lambda x: x.cuda(), input)
    mixed, clean, seq_len = input[0].cuda(), input[1].cuda(), input[2]
    bsz, nCH, Tt = mixed.size()
    mixedSTFT = stft(mixed.view(bsz*nCH, Tt))
    Tf = mixedSTFT.size(2)
    cleanSTFT = None
    if(not Tf % stride_product == 1):
        nPad_time = stride_product*math.ceil(Tf/stride_product) - Tf + 1
        mixedSTFT = F.pad(mixedSTFT, (0, 0, 0, nPad_time, 0, 0))  # (Fs,Fe,Ts,Te, real, imag)
        cleanSTFT = stft(clean)
        cleanSTFT = F.pad(cleanSTFT, (0, 0, 0, nPad_time, 0, 0))
        clean_real, clean_imag = cleanSTFT[..., 0], cleanSTFT[..., 1]
        clean = istft(clean_real.squeeze(), clean_imag.squeeze(), clean.size(-1))
    mixedSTFT = mixedSTFT.view(bsz, nCH, mixedSTFT.size(1), mixedSTFT.size(2), 2) # NxnCHxFxTfx2 (real/imag)
    mixed_real, mixed_imag = mixedSTFT[..., 0], mixedSTFT[..., 1]
    if(not loss_type == 'cossim_mag'):
        out_real, out_imag = net(mixed_real, mixed_imag)
        out_audio = istft(out_real.squeeze(), out_imag.squeeze(), mixed.size(-1)) # ver1 --> len(out) ~= len(clean) for dt/et
    else:
        out_mag = net(mixed_real, mixed_imag)
        noisy_phase_CH1 = torch.atan2(mixed_real[:, 0], mixed_imag[:, 0])
        out_real = out_mag*torch.cos(noisy_phase_CH1)
        out_imag = out_mag*torch.sin(noisy_phase_CH1)
        out_audio = istft(out_real.squeeze(), out_imag.squeeze(), mixed.size(-1))  # ver1 --> len(out) ~= len(clean) for dt/et

    for i, l in enumerate(seq_len): # zero padding to output audio
        out_audio[i, l:] = 0

    if (loss_type == 'cossim_spec'):
        if(cleanSTFT is None):
            cleanSTFT = stft(clean)
            clean_real, clean_imag = cleanSTFT[..., 0], cleanSTFT[..., 1]
        loss = Loss(clean_real, clean_imag, out_real, out_imag)
    elif (loss_type == 'sInvSDR_time'):
        loss = -Loss(clean, out_audio)  # loss should be opposite direcetion of SDR
    elif(loss_type == 'cossim_time'):
        loss = Loss(clean, out_audio)
    elif(loss_type == 'cossim_mag'):
        if (cleanSTFT is None):
            cleanSTFT = stft(clean)
            clean_real, clean_imag = cleanSTFT[..., 0], cleanSTFT[..., 1]
        clean_mag = torch.sqrt(clean_real*clean_real + clean_imag*clean_imag)
        loss = Loss(clean_mag, out_mag)
        #pdb.set_trace()


    if(write_wav == 'train'):
        if(not all_wav):
            librosa.output.write_wav('wavs/' + str(expnum) + '/mixed_' + data_type + '.wav', mixed[0][0].cpu().data.numpy()[:seq_len[0].item()], 16000, norm=True) # CH1
            librosa.output.write_wav('wavs/' + str(expnum) + '/clean_' + data_type + '.wav', clean[0].cpu().data.numpy()[:seq_len[0].item()], 16000, norm=True)
            librosa.output.write_wav('wavs/' + str(expnum) + '/out_' + data_type + '.wav', out_audio[0].cpu().data.numpy()[:seq_len[0].item()], 16000, norm=True)
        else:
            for n in range(bsz):
                librosa.output.write_wav('wavs/' + str(expnum) + '/mixed_' + data_type + '_' + str(n) + '.wav', mixed[n][0].cpu().data.numpy()[:seq_len[n].item()], 16000, norm=True)  # CH1
                librosa.output.write_wav('wavs/' + str(expnum) + '/clean_' + data_type + '_' + str(n) + '.wav', clean[n].cpu().data.numpy()[:seq_len[n].item()], 16000, norm=True)
                librosa.output.write_wav('wavs/' + str(expnum) + '/out_' + data_type + '_' + str(n) + '.wav', out_audio[n].cpu().data.numpy()[:seq_len[n].item()], 16000, norm=True)
    elif(write_wav == 'test'):
        reverb_path = input[3]
        for n in range(bsz):
            reverb_path_split = reverb_path[n].split('/')
            id = reverb_path_split[-1]
            reverb_dir = 'wavs/' + str(expnum) + '_eval/' + '/'.join(reverb_path[n].split('/')[-5:-1])
            if not os.path.exists(reverb_dir):
                os.makedirs(reverb_dir)
            librosa.output.write_wav(reverb_dir + '/' + id + '_ch1.wav', out_audio[n].cpu().data.numpy()[:seq_len[n].item()], 16000) # this format follows official evaluation script for SIMDATA-dt
                                                                                                                                    # but why _ch1.wav as output?

    if (Eval is None): # train.py
        return loss
    else: # test.py
        # without delay distortion allowed
        if(eval_type == 'time'):
            eval_metric = Eval(clean, out_audio)
        elif(eval_type == 'spec'):
            eval_metric = Eval(clean_real, clean_imag, out_real, out_imag)
        elif (eval_type == 'from_loss'):
            eval_metric = Eval(loss)

        # with delay distortion allowed
        if (eval_type == 'time'):
            eval_metric_multitime = torch.FloatTensor(bsz, 20)
            for i in range(-10, 11):
                if(i < 0): # negative time delay --> pad at front part
                    out_audio_shifted = F.pad(out_audio[..., :i], (-i, 0, 0, 0))
                    eval_metric_multitime[:, i + 10] = Eval(clean, out_audio_shifted)
                elif(i > 0):  # positive time delay --> pad at back part
                    out_audio_shifted = F.pad(out_audio[..., i:], (0, i, 0, 0))
                    eval_metric_multitime[:, i + 9] = Eval(clean, out_audio_shifted)

            torch.save(eval_metric_multitime, 'wavs/' + str(expnum) + '_eval/eval_metric_multitime_' + data_type + '.pth')

            eval_metric_multitime_min = torch.min(eval_metric_multitime, dim=1)[0]
            eval_metric_multitime_max = torch.max(eval_metric_multitime, dim=1)[0]
        elif (eval_type == 'from_loss'): # not supported so output 0
            eval_metric_multitime_min = torch.FloatTensor([0])
            eval_metric_multitime_max = torch.FloatTensor([0])
        else:
            eval_metric_multitime_min = 0
            eval_metric_multitime_max = 0


        return [loss, eval_metric, eval_metric_multitime_max, eval_metric_multitime_min]
