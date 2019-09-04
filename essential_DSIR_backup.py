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

def forward_common(istft, input, net, Loss, data_type, loss_type, stride_product, mode='train', expnum=-1, save_wav=False, Eval=None, fix_len_by_cl='input', count=0, w_var=0, f_start_ratio=0, f_end_ratio=1):
    eps = 1e-10
    mixedSTFT, cleanSTFT, len_STFT_cl = input[0].cuda(), input[1].cuda(), input[2]
    bsz, nCH, F, Tf, _ = mixedSTFT.size()

    if(f_end_ratio < 1):
        K = cleanSTFT.size(1)
        k_start = int(K * f_start_ratio)
        k_end = int(K * f_end_ratio)
        cleanSTFT = cleanSTFT[:, k_start:k_end + 1, :]

    if(stride_product > 0 and not Tf % stride_product == 1):
        nPad_time = stride_product*math.ceil(Tf/stride_product) - Tf + 1
        mixedSTFT = Func.pad(mixedSTFT, (0, 0, 0, nPad_time, 0, 0))  # (Fs,Fe,Ts,Te, real, imag)
        #pdb.set_trace()
        cleanSTFT = Func.pad(cleanSTFT, (0, 0, 0, nPad_time, 0, 0))

    clean_real, clean_imag = cleanSTFT[..., 0], cleanSTFT[..., 1]
    mixed_real, mixed_imag = mixedSTFT[..., 0], mixedSTFT[..., 1]

    out_real, out_imag, mask1, mask2 = net(mixed_real, mixed_imag) #mask1&2 can be either real/imag or mag/phs

    if(fix_len_by_cl == 'eval'):
        out_real = out_real[:, :, :clean_real.size(2)]
        out_imag = out_imag[:, :, :clean_imag.size(2)]

    for i, l in enumerate(len_STFT_cl): # zero padding to output audio
        out_real[i, :, l:] = 0
        out_imag[i, :, l:] = 0

    loss = -Loss(clean_real, clean_imag, out_real, out_imag)

    if(mode == 'train'):
        if(save_wav):
            mixed_time, clean_time, len_time = input[3], input[4], input[5]
            if(loss_type == 'sInvSDR_mag' or loss_type == 'cossim_mag'):
                noisy_phase_CH1 = torch.atan2(mixed_real[:, 0] + eps, mixed_imag[:, 0])  # eps for preventing divide by zero
                out_mag = torch.sqrt(out_real*out_real + out_imag*out_imag)
                if (fix_len_by_cl == 'eval'):
                    noisy_phase_CH1 = noisy_phase_CH1[..., :clean_real.size(2)]
                out_audio_real = out_mag * torch.cos(noisy_phase_CH1)
                out_audio_imag = out_mag * torch.sin(noisy_phase_CH1)
            else:
                out_audio_real = out_real
                out_audio_imag = out_imag
            out_audio = istft(out_audio_real.squeeze(), out_audio_imag.squeeze(), mixed_time.size(-1))
            for i, l in enumerate(len_time):  # zero padding to output audio
                out_audio[i, l:] = 0

            T0 = len_time[0].item()
            # librosa
            #librosa.output.write_wav('wavs/' + str(expnum) + '/mixed_' + data_type + '.wav', mixed_time[0][0].data.numpy()[:T0], 16000, norm=True) # CH1
            #librosa.output.write_wav('wavs/' + str(expnum) + '/clean_' + data_type + '.wav', clean_time[0].data.numpy()[:T0], 16000, norm=True)
            #librosa.output.write_wav('wavs/' + str(expnum) + '/out_' + data_type + '.wav', out_audio[0].cpu().data.numpy()[:T0], 16000, norm=True)

            # torchaudio
            #torchaudio.save('wavs/' + str(expnum) + '/mixed_' + data_type + '.wav', normalize(mixed_time[0][0][:T0]).cpu(), 16000) # CH1
            #torchaudio.save('wavs/' + str(expnum) + '/clean_' + data_type + '.wav', normalize(clean_time[0][:T0]).cpu(), 16000)
            #torchaudio.save('wavs/' + str(expnum) + '/out_' + data_type + '.wav', normalize(out_audio[0][:T0]).cpu(), 16000)

            # soundfile
            sf.write('wavs/' + str(expnum) + '/mixed_' + data_type + '.wav', normalize(mixed_time[0][0][:T0]).data.cpu().numpy(), 16000) # CH1
            sf.write('wavs/' + str(expnum) + '/clean_' + data_type + '.wav', normalize(clean_time[0][:T0]).data.cpu().numpy(), 16000)
            sf.write('wavs/' + str(expnum) + '/out_' + data_type + '.wav', normalize(out_audio[0][:T0]).data.cpu().numpy(), 16000)


    elif(mode == 'test'):
        reverb_path = input[3]
        for n in range(bsz):
            reverb_path_split = reverb_path[n].split('/')
            id = reverb_path_split[-1]
            reverb_dir = 'wavs/' + str(expnum) + '_eval/' + '/'.join(reverb_path[n].split('/')[-5:-1])
            if not os.path.exists(reverb_dir):
                os.makedirs(reverb_dir)
            #librosa.output.write_wav(reverb_dir + '/' + id + '_ch1.wav', out_audio[n].cpu().data.numpy()[:seq_len[n].item()], 16000) # this format follows official evaluation script for SIMDATA-dt
            #torchaudio.save(reverb_dir + '/' + id + '_ch1.wav', normalize(out_audio[n][:seq_len[n].item()]).cpu(), 16000)  # this format follows official evaluation script for SIMDATA-dt
            sf.write(reverb_dir + '/' + id + '_ch1.wav', normalize(out_audio[n][:seq_len[n].item()]).data.cpu().numpy(), 16000)
    elif(mode == 'generate'): # generate spectroram
        specs_path = 'specs/' + str(expnum) + '/' + data_type + '_' + str(count) + '.mat'
        sio.savemat(specs_path, {'mixed_real':mixed_real.data.cpu().numpy(), 'mixed_imag':mixed_imag.data.cpu().numpy(),
                                 'out_real': out_real.data.cpu().numpy(), 'out_imag':out_imag.data.cpu().numpy(),
                                 'clean_real': clean_real.data.cpu().numpy(), 'clean_imag':clean_imag.data.cpu().numpy(),
                                 'mask_real':mask1.data.cpu().numpy(), 'mask_imag':mask2.data.cpu().numpy()})

    if(Loss == Eval):
        eval_metric = -loss
    else:
        eval_metric = Eval(clean_real, clean_imag, out_real, out_imag)

    if (w_var > 0):
        l_var_wr = var_time(mask1)
        l_var_wi = var_time(mask2)
        l_var_w = l_var_wr + l_var_wi
        loss = [loss, w_var*l_var_w]

    return loss, eval_metric


def eval_segment(interval_list, input, net, Eval, stride_product, fix_len_by_cl):
    nInterval = len(interval_list)-1
    eval_metric_interval = torch.zeros(nInterval, 1)
    mixedSTFT, cleanSTFT, len_STFT_cl = input[0].cuda(), input[1].cuda(), input[2]
    bsz, nCH, F, Tf, _ = mixedSTFT.size()

    if(stride_product > 0):
        if (not Tf % stride_product == 1):
            nPad_time = stride_product * math.ceil(Tf / stride_product) - Tf + 1
            mixedSTFT = Func.pad(mixedSTFT, (0, 0, 0, nPad_time, 0, 0))  # (Fs,Fe,Ts,Te, real, imag)
            cleanSTFT = Func.pad(cleanSTFT, (0, 0, 0, nPad_time, 0, 0))

    clean_real, clean_imag = cleanSTFT[..., 0], cleanSTFT[..., 1]
    mixed_real, mixed_imag = mixedSTFT[..., 0], mixedSTFT[..., 1]
    out_real, out_imag, mask1, mask2 = net(mixed_real, mixed_imag)  # mask1&2 can be either real/imag or mag/phs

    if (fix_len_by_cl == 'eval'):
        out_real = out_real[:, :, :clean_real.size(2)]
        out_imag = out_imag[:, :, :clean_imag.size(2)]

    for i, l in enumerate(len_STFT_cl):  # zero padding to output audio
        out_real[i, :, l:] = 0
        out_imag[i, :, l:] = 0

    for n in range(nInterval):
        f_start = interval_list[n]
        f_end = interval_list[n+1]-1
        #pdb.set_trace()
        clean_real_segment = clean_real[:, f_start:f_end, :]
        clean_imag_segment = clean_imag[:, f_start:f_end, :]
        out_real_segment = out_real[:, f_start:f_end, :]
        out_imag_segment = out_imag[:, f_start:f_end, :]

        eval_metric = Eval(clean_real_segment, clean_imag_segment, out_real_segment, out_imag_segment)

        eval_metric_interval[n] = eval_metric.mean().item()

    return eval_metric_interval


def measure_reverb_loss(loader, Loss):
    bar = tqdm(loader)
    loss_total = 0
    with torch.no_grad():
        for input in bar:
            mixedSTFT, cleanSTFT = input[0], input[1].cuda()
            mixedSTFT_CH1 = mixedSTFT[:, 0].squeeze().cuda()

            clean_real, clean_imag = cleanSTFT[..., 0], cleanSTFT[..., 1]
            mixed_real, mixed_imag = mixedSTFT_CH1[..., 0], mixedSTFT_CH1[..., 1]

            loss = -Loss(clean_real, clean_imag, mixed_real, mixed_imag)  # loss_type = sInvSDR_spec
            loss = torch.mean(loss)

            loss_total += loss.item()

    loss_total = loss_total/len(loader)

    return loss_total