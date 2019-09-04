import os

import torch
import torch.nn.functional as Func
#import librosa
#import torchaudio
import soundfile as sf
import math
import scipy.io as sio
from models.loss import var_time
import random

import pdb

def normalize(y):
    y = y/max(abs(y))
    return y

def forward_common(istft, input, net, Loss, data_type, loss_type, stride_product, mode='train', expnum=-1, save_wav=False, Eval=None, fix_len_by_cl='input', count=0, w_var=0, f_start_ratio=0, f_end_ratio=1):
    eps = 1e-10
    mixedSTFT, cleanSTFT, len_STFT_cl = input[0].cuda(), input[1].cuda(), input[2]

    if(mixedSTFT.dim() == 4): # for singleCH experiment
        mixedSTFT = mixedSTFT.unsqueeze(1)
    bsz, nCH, F, Tf, _ = mixedSTFT.size()

    if(f_end_ratio < 1):
        K = cleanSTFT.size(1)
        k_start = int(K * f_start_ratio)
        k_end = int(K * f_end_ratio)
        cleanSTFT = cleanSTFT[:, k_start:k_end + 1, :]

    if(stride_product > 0 and not Tf % stride_product == 1):
        nPad_time = stride_product*math.ceil(Tf/stride_product) - Tf + 1
        mixedSTFT = Func.pad(mixedSTFT, (0, 0, 0, nPad_time, 0, 0))  # (Fs,Fe,Ts,Te, real, imag)
        cleanSTFT = Func.pad(cleanSTFT, (0, 0, 0, nPad_time, 0, 0))

    clean_real, clean_imag = cleanSTFT[..., 0], cleanSTFT[..., 1]
    mixed_real, mixed_imag = mixedSTFT[..., 0], mixedSTFT[..., 1]

    out_list, mask_list = net(mixed_real, mixed_imag)

    if(fix_len_by_cl == 'eval'):
        Tmax_rev = out_list[0].size(-1)
        Tmax_cl = clean_real.size(-1)

        if(Tmax_cl < Tmax_rev):
            for n in range(len(out_list)):
                out_list[n] = out_list[n][:, :, :clean_real.size(-1)]
        elif(Tmax_cl > Tmax_rev):
            clean_real = clean_real[:, :, :Tmax_rev]
            clean_imag = clean_imag[:, :, :Tmax_rev]
            Tmax_cl = Tmax_rev

    for n in range(len(out_list)):
        for i, l in enumerate(len_STFT_cl): # zero padding to output audio
            out_list[n][i, :, min(l, Tmax_cl):] = 0

    loss = -Loss(clean_real, clean_imag, out_list)

    if(mode == 'train'):
        if(save_wav):
            mixed_time, clean_time, len_time = input[3], input[4], input[5]
            if(loss_type == 'sInvSDR_mag' or loss_type == 'cossim_mag'):
                noisy_phase_CH1 = torch.atan2(mixed_real[:, 0] + eps, mixed_imag[:, 0])  # eps for preventing divide by zero
                if(len(out_list) == 2):
                    out_mag = torch.sqrt(out_list[0]*out_list[0] + out_list[1]*out_list[1])
                elif(len(out_list) == 1):
                    out_mag = torch.sqrt(out_list[0])
                if (fix_len_by_cl == 'eval'):
                    noisy_phase_CH1 = noisy_phase_CH1[..., :clean_real.size(2)]
                out_audio_real = out_mag * torch.cos(noisy_phase_CH1)
                out_audio_imag = out_mag * torch.sin(noisy_phase_CH1)
            else:
                out_audio_real = out_list[0]
                out_audio_imag = out_list[1]
            out_audio = istft(out_audio_real.squeeze(), out_audio_imag.squeeze(), mixed_time.size(-1))
            for i, l in enumerate(len_time):  # zero padding to output audio
                out_audio[i, l:] = 0

            T0 = len_time[0].item()

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
            sf.write(reverb_dir + '/' + id + '_ch1.wav', normalize(out_audio[n][:seq_len[n].item()]).data.cpu().numpy(), 16000)

    elif(mode == 'generate'): # generate spectroram
        specs_path = 'specs/' + str(expnum) + '/' + data_type + '_' + str(count) + '.mat'
        #pdb.set_trace()
        sio.savemat(specs_path, {'mixed_real':mixed_real.data.cpu().numpy(), 'mixed_imag':mixed_imag.data.cpu().numpy(),
                                 'out_real': out_list[0].data.cpu().numpy(), 'out_imag':out_list[1].data.cpu().numpy(),
                                 'clean_real': clean_real.data.cpu().numpy(), 'clean_imag':clean_imag.data.cpu().numpy(),
                                 'mask_real':mask_list[0].data.cpu().numpy(), 'mask_imag':mask_list[1].data.cpu().numpy()})

    if(Loss == Eval):
        eval_metric = -loss
    else:
        eval_metric = Eval(clean_real, clean_imag, out_real, out_imag)

    if (w_var > 0):
        l_var_w = 0
        for m in range(len(mask_list)):
            l_var_w += var_time(mask_list[m])
        '''
        l_var_wr = var_time(mask1)
        l_var_wi = var_time(mask2)
        l_var_w = l_var_wr + l_var_wi
        '''
        loss = [loss, w_var*l_var_w]

    return loss, eval_metric


def forward_WER(istft, stft_AM, input, net, netAM, STFT_to_LMFB, Loss, decoding_type, stride_product, decoder, logFile, fix_len_by_cl = 'eval', expnum=-1):
    mixedSTFT, cleanSTFT, len_STFT_cl, mixed_time, len_time = input[0].cuda(), input[1].cuda(), input[2], input[3], input[5]

    if(mixedSTFT.dim() == 4): # for singleCH experiment
        mixedSTFT = mixedSTFT.unsqueeze(1)
    bsz, nCH, F, Tf, _ = mixedSTFT.size()

    if(stride_product > 0 and not Tf % stride_product == 1):
        nPad_time = stride_product*math.ceil(Tf/stride_product) - Tf + 1
        mixedSTFT = Func.pad(mixedSTFT, (0, 0, 0, nPad_time, 0, 0))  # (Fs,Fe,Ts,Te, real, imag)
        cleanSTFT = Func.pad(cleanSTFT, (0, 0, 0, nPad_time, 0, 0))

    clean_real, clean_imag = cleanSTFT[..., 0], cleanSTFT[..., 1]
    mixed_real, mixed_imag = mixedSTFT[..., 0], mixedSTFT[..., 1]

    out_list, mask_list = net(mixed_real, mixed_imag)

    for n in range(len(out_list)): # fix_len_by_cl == eval
        out_list[n] = out_list[n][:, :, :clean_real.size(-1)]

    if(fix_len_by_cl == 'eval'):
        Tmax_rev = out_list[0].size(-1)
        Tmax_cl = clean_real.size(-1)

        if(Tmax_cl < Tmax_rev):
            for n in range(len(out_list)):
                out_list[n] = out_list[n][:, :, :clean_real.size(-1)]
        elif(Tmax_cl > Tmax_rev):
            clean_real = clean_real[:, :, :Tmax_rev]
            clean_imag = clean_imag[:, :, :Tmax_rev]
            Tmax_cl = Tmax_rev

    for n in range(len(out_list)):
        for i, l in enumerate(len_STFT_cl): # zero padding to output audio
            out_list[n][i, :, min(l, Tmax_cl):] = 0

    loss = -Loss(clean_real, clean_imag, out_list)
    #pdb.set_trace()
    loss = loss.mean().item() # we don't have to store computation graph unless we backprop from loss

    out_real = out_list[0]
    out_imag = out_list[1]
    out_audio = istft(out_real.squeeze(), out_imag.squeeze(), mixed_time.size(-1))
    for i, l in enumerate(len_time):  # zero padding to output audio
        out_audio[i, l:] = 0

    # STFT (ASR setting)
    enh_STFT = stft_AM(out_audio)

    # STFT to LMFB
    enh_STFT_real = enh_STFT[..., 0]
    enh_STFT_imag = enh_STFT[..., 1]
    enh_LMFB = STFT_to_LMFB(enh_STFT_real, enh_STFT_imag)

    # Decoding
    texts = input[6]
    N = enh_LMFB.size(0)
    input_percentages = torch.FloatTensor(N).fill_(1) # let's regard every utterance have same size (audio filled with zero = silence)
    target_sizes = torch.IntTensor(N)
    targets = []
    #pdb.set_trace()
    for n in range(N):
        text = texts[n]
        target_sizes[n] = len(text)
        targets.extend(text)
    targets = torch.IntTensor(targets)

    if(decoding_type == 'greedy'):
        wer, cer, nWord, nChar = greedy_decoding(enh_LMFB, netAM, targets, input_percentages, target_sizes, decoder, logFile)

    elif(decoding_type == 'beam_LM'):
        prob = ASR(enh_LMFB)
        prob = prob.transpose(0, 1)
        sizes = input_percentages.mul_(int(seq_length)).int()

        # Save logits
        logits.append((prob.data.cpu().numpy(), sizes.numpy()))
        logits_path = str(expnum) + '/logits.npy'  # Assume each experiment folder perform one decoding
        print('save ' + logits_path)
        np.save(logits_path, logits)

    return wer, cer, loss

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


def greedy_decoding(enhanced, netAM, targets, input_percentages, target_sizes, decoder, logFile, transcript_prob=0.001):
    # unflatten targets
    split_targets = []
    offset = 0
    for size in target_sizes:
        split_targets.append(targets[offset:offset + size])
        offset += size

    # step 1) Decoding to get wer & cer
    prob = netAM(enhanced)
    prob = prob.transpose(0,1)
    T = prob.size(0)
    sizes = input_percentages.mul_(int(T)).int()

    decoded_output, _ = decoder.decode(prob.data, sizes)
    target_strings = decoder.convert_to_strings(split_targets)
    we, ce, total_word, total_char = 0, 0, 0, 0

    for x in range(len(target_strings)):
        decoding, reference = decoded_output[x][0], target_strings[x][0]
        nChar = len(reference)
        nWord = len(reference.split())
        we_i = decoder.wer(decoding, reference)
        ce_i = decoder.cer(decoding, reference)
        we += we_i
        ce += ce_i
        total_word += nWord
        total_char += nChar
        if (random.uniform(0, 1) < transcript_prob):
            print('reference = ' + reference)
            print('decoding = ' + decoding)
            print('wer = ' + str(we_i/float(nWord)) + ', cer = ' + str(ce_i/float(nChar)))
        logFile.write('reference = ' + reference + '\n')
        logFile.write('decoding = ' + decoding + '\n')
        logFile.write('wer = ' + str(we_i/float(nWord)) + ', cer = ' + str(ce_i/float(nChar)) + '\n')

    wer = we/total_word
    cer = ce/total_char

    return wer, cer, total_word, total_char