import os
import sys
import torch
import numpy as np

import torch.optim as optim

from tqdm import tqdm
import gc

import utils
from utils import get_stride_product_time, count_parameters
#from models.layers.istft import ISTFT
from models.loss import cossim_time, cossim_spec, cossim_mag, sInvSDR_time, sInvSDR_spec, negative_MSE, sInvSDR_mag, \
    srcIndepSDR_mag, srcIndepSDR_freqpower, srcIndepSDR_mag_diffperT, srcIndepSDR_freqpower_diffperT, srcIndepSDR_freqpower_by_enhanced, \
    srcIndepSDR_Cproj_by_WH, srcIndepSDR_Cproj_by_SShat

from se_dataset import SpecDataset
from torch.utils.data import DataLoader
import scipy.io as sio
#from memprof import *

import pdb

from essential import forward_common
from config import get_config


# memory leakage debug
'''
import tracemalloc
snapshot = None
def trace_print(logfile):
    global snapshot
    snapshot2 = tracemalloc.take_snapshot()
    snapshot2 = snapshot2.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
        tracemalloc.Filter(False, "<unknown>"),
        tracemalloc.Filter(False, tracemalloc.__file__)
    ))

    if snapshot is not None:
        if(logfile is not None):
            logfile.write("================================== Begin Trace: \n")
        print("================================== Begin Trace:")
        top_stats = snapshot2.compare_to(snapshot, 'lineno', cumulative=True)
        for stat in top_stats[:100]:
            #pdb.set_trace()
            #logfile.write(stat)
            if (logfile is not None):
                if(stat.count > 0):
                    info = stat.traceback[0].filename + ':' + str(stat.traceback[0].lineno) + \
                           ' size=' + str(stat.size/1024) + ' KiB (' + str(stat.size_diff/1024) + '), count=' + str(stat.count) + ' (' + str(stat.count_diff) + '), average=' + str(stat.size/stat.count/1024) + ' KiB'
                else:
                    info = stat.traceback[0].filename + ':' + str(stat.traceback[0].lineno) + \
                           ' size=' + str(stat.size/1024) + ' KiB (' + str(stat.size_diff/1024) + '), count=' + str(stat.count) + ' (' + str(stat.count_diff) + ')'

                logfile.write(info + '\n')
            print(stat)
    snapshot = snapshot2
'''
#@memprof(threshold = 10240)
def main(args):
    #print('!!!!!!!!!!!!!!!!! USING MEM DEBUG FILE !!!!!!!!!!!!!!!!!!!!')
    #mem_debug_file = open('mem_debug_file_' + str(args.expnum) + '.txt', 'w')
    #tracemalloc.start()
    assert (args.expnum >= 1)
    if (args.mic_sampling == 'ref_manual'):
        args.subset1 = args.subset1.split(',')  # string to list
        args.subset2 = args.subset2.split(',')  # string to list

    # Get free gpu
    gpuIdx = utils.get_freer_gpu()
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpuIdx)
    print('gpuIdx = ' + str(gpuIdx) + ' is selected')

    prefix = 'data_sorted/'
    #pdb.set_trace()
    if(not args.mode == 'generate'): # for generation mode, please manually write manifest
        if(args.grid_cm > 0):
            if(args.grid_cm == 1):
                # new data (large room, fixed mic)
                #pdb.set_trace()
                if(len(args.tr_manifest) == 0):
                    args.tr_manifest = prefix + 'L553_fixedmic_' + args.directivity + '_' + str(args.grid_cm) + 'cm' + '_RT0.2_tr.csv'
                if (len(args.val_manifest) == 0):
                    args.val_manifest = prefix + 'L553_fixedmic_' + args.directivity + '_' + str(args.grid_cm) + 'cm' + '_RT0.2_val.csv'
                '''
                if (len(args.te1_manifest) == 0):
                    args.te1_manifest = prefix + 'L553_fixedmic_' + args.directivity + '_' + str(args.grid_cm) + 'cm' + '_RT0.2_te1.csv'
                if(len(args.te2_manifest) == 0):
                    args.te2_manifest = prefix + 'L553_fixedmic_randomsrc_' + args.directivity + '_' + str(args.grid_cm) + 'cm' + '_RT0.2_te2.csv'
                '''
                # old data (too small room, mic not fixed)
                '''
                args.tr_manifest = prefix + 'reverb_L221_fixedrange_1cm_RT0.25_IR+Src_tr.csv'
                args.val_manifest = prefix + 'reverb_L221_fixedrange_1cm_RT0.25_IR+Src_val.csv'
                args.te1_manifest = prefix + 'reverb_L221_fixedrange_1cm_RT0.25_IR+Src_te1.csv'
                args.te2_manifest = prefix + 'reverb_L221_randomrange_1cm_RT0.25_IR+Src_te2.csv'
                '''
                #args.load_IR = True # True by default
            else:
                args.tr_manifest = prefix + 'L221_10105_grid' + str(args.grid_cm) + 'cm' + '_tr_exists_only.csv'
                args.val_manifest = prefix + 'L221_10105_grid10cm_val_1per.csv' # mini valid set
                #args.te1_manifest = prefix + 'L221_10105_grid' + str(args.grid_cm) + 'cm' + '_te.csv'
                args.te1_manifest = ''

        if(args.mode == 'RT_analysis'):
            args.tr_manifest = prefix + 'reverb_tr_simu_8ch_RT60.csv'
            args.val_manifest = prefix + 'reverb_dt_simu_8ch_RT60.csv'
            args.te1_manifest = ''

        #args.val_trainIR_manifest = prefix + 'reverb_dt_simu_8ch_trainIR_paired.csv'

        if(args.REV_VIS):
            args.tr_manifest = prefix + 'RIV-VIS_tr.csv'
            #args.tr_manifest = prefix + 'visualize_scenario_augmented_tr.csv'
            #args.val_manifest = prefix + 'RIV-VIS_val.csv' # use original reverb dataset
            args.te1_manifest = prefix + 'visualize_scenario_augmented_te_smallset.csv'

    n_fft = args.nFFT
    win_size = args.nWin
    if(args.hop_length == 0):
        hop_length = int(win_size/2)
    else:
        hop_length = args.hop_length
    #print('1')
    #
    window_path = 'window_' + str(win_size) + '.pth'
    if not os.path.exists(window_path):
        window = torch.hann_window(win_size)
        torch.save(window, window_path)
    else:
        window = torch.load(window_path, map_location=torch.device('cpu'))
    window = window.cuda()

    #print('2')
    f_start_ratio = args.f_start / (args.fs / 2)
    f_end_ratio = args.f_end / (args.fs / 2)

    #loss_w_var = 0 #dummy variable

    stft = lambda x: torch.stft(x, n_fft, hop_length, win_length = win_size, window=window)
    #print('3')
    if(args.use_ISTFT):
        istft = ISTFT(n_fft, hop_length, window='hanning').cuda()
    else:
        istft = None

    if(args.mode == 'generate'):
        args.return_path = True
        shuffle_train_loader = False
    else:
        shuffle_train_loader = True

    src_range_list = args.src_range.replace("'", "").split(',')
    src_range_list = [float(p) for p in src_range_list] # convert str to float
    assert (len(src_range_list) == 6)


    #utils.CPUmemDebug('before dataset init',mem_debug_file)
    if (len(args.tr_manifest) > 0):
        train_dataset = SpecDataset(manifest_path=args.tr_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl, return_path=args.return_path,
                                    load_IR=args.load_IR, use_localization=args.use_localization, src_range=src_range_list, nSource=args.nSource,
                                    start_ratio=args.start_ratio, end_ratio=args.end_ratio,
                                    clamp_frame=args.clamp_frame, ref_mic_direct_td_subtract=args.ref_mic_direct_td_subtract,
                                    interval_cm=args.interval_cm_tr) # start_ratio, end_ratio only for training dataset
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate, shuffle=shuffle_train_loader, num_workers=0)

    if (len(args.val_manifest) > 0):
        val_dataset = SpecDataset(manifest_path=args.val_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl, return_path=args.return_path,
                                  load_IR=args.load_IR, use_localization=args.use_localization, src_range='all', nSource=args.nSource,
                                  clamp_frame=args.clamp_frame, ref_mic_direct_td_subtract=args.ref_mic_direct_td_subtract,
                                  interval_cm=args.interval_cm_te)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate, shuffle=False, num_workers=0)

    if(len(args.te1_manifest) > 0):
        test1_dataset = SpecDataset(manifest_path=args.te1_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl,
                                    load_IR=args.load_IR, use_localization=args.use_localization, src_range='all', nSource=args.nSource,
                                    clamp_frame=args.clamp_frame, ref_mic_direct_td_subtract=args.ref_mic_direct_td_subtract,
                                    interval_cm=args.interval_cm_te)
        test1_loader = DataLoader(dataset=test1_dataset, batch_size=args.batch_size, collate_fn=test1_dataset.collate, shuffle=False, num_workers=0)

    if(len(args.te2_manifest) > 0):
        test2_dataset = SpecDataset(manifest_path=args.te2_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl,
                                    load_IR=args.load_IR, use_localization=args.use_localization, src_range='all', nSource=args.nSource,
                                    clamp_frame=args.clamp_frame, ref_mic_direct_td_subtract=args.ref_mic_direct_td_subtract,
                                    interval_cm=args.interval_cm_te) # for test2, set pos_range as 'all' (all positions within a room)
        test2_loader = DataLoader(dataset=test2_dataset, batch_size=args.batch_size, collate_fn=test2_dataset.collate, shuffle=False, num_workers=0)

    if(args.eval_iter == 0):
        args.eval_iter = len(train_loader)

    torch.set_printoptions(precision=10, profile="full")
    #utils.CPUmemDebug('after dataset init', mem_debug_file)

    # Set loss type
    if(args.loss_type == 'cossim_time'):
        Loss = cossim_time
    elif(args.loss_type == 'cossim_spec'):
        Loss = cossim_spec
    elif(args.loss_type == 'cossim_mag'):
        Loss = cossim_mag
    elif(args.loss_type == 'sInvSDR_time'):
        Loss = sInvSDR_time
    elif (args.loss_type == 'sInvSDR_spec'):
        Loss = sInvSDR_spec
    elif (args.loss_type == 'sInvSDR_mag'):
        Loss = sInvSDR_mag
    elif(args.loss_type == 'srcIndepSDR_mag'):
        Loss = srcIndepSDR_mag
    elif(args.loss_type == 'srcIndepSDR_Cproj_by_WH'):
        Loss = srcIndepSDR_Cproj_by_WH
    elif(args.loss_type == 'srcIndepSDR_Cproj_by_SShat'):
        #Loss = srcIndepSDR_Cproj_by_SShat
        print('srcIndepSDR_Cproj_by_SShat, eps = ' + str(args.eps))
        Loss = lambda clean_real, clean_imag, out_real, out_imag, Tlist: srcIndepSDR_Cproj_by_SShat(clean_real, clean_imag, out_real, out_imag, Tlist, args.eps)

    elif(args.loss_type == 'srcIndepSDR_freqpower'):
        weights_per_freq = torch.load('weights_per_freq_' + str(args.nFFT) + '.pth').cuda()
        Pf = weights_per_freq*weights_per_freq
        Pf_sum = Pf.sum()
        Pf = Pf/Pf_sum # normalize

        Loss = lambda Wreal, Wimag, Hreal, Himag, Tlist: srcIndepSDR_freqpower(Wreal, Wimag, Hreal, Himag, Tlist, Pf)
    elif (args.loss_type == 'srcIndepSDR_mag_diffperT'):
        Loss = srcIndepSDR_mag_diffperT
    elif (args.loss_type == 'srcIndepSDR_freqpower_diffperT'):
        weights_per_freq = torch.load('weights_per_freq_' + str(args.nFFT) + '.pth').cuda()
        Pf = weights_per_freq * weights_per_freq
        Pf_sum = Pf.sum()
        Pf = Pf / Pf_sum  # normalize

        Loss = lambda Wreal, Wimag, Hreal, Himag, Tlist: srcIndepSDR_freqpower_diffperT(Wreal, Wimag, Hreal, Himag, Tlist, Pf)

    elif(args.loss_type == 'negative_MSE'):
        Loss = negative_MSE

    elif(args.loss_type == 'srcIndepSDR_freqpower_by_enhanced'):
        weights_per_freq = torch.load('weights_per_freq_' + str(args.nFFT) + '.pth').cuda()
        Pf = weights_per_freq * weights_per_freq
        Pf_sum = Pf.sum()
        Pf = Pf / Pf_sum  # normalize

        Loss = lambda out_real, out_imag, Tlist: srcIndepSDR_freqpower_by_enhanced(out_real, out_imag, Tlist, Pf)

    if(args.eval_type == 'sInvSDR_time'):
        Eval = sInvSDR_time
    elif(args.eval_type == 'sInvSDR_spec'):
        Eval = sInvSDR_spec
    elif(args.eval_type == 'sInvSDR_mag'):
        Eval = sInvSDR_mag
    elif(args.eval_type == 'srcIndepSDR_mag'):
        Eval = srcIndepSDR_mag
    elif (args.eval_type == 'srcIndepSDR_freqpower'):
        weights_per_freq = torch.load('weights_per_freq_' + str(args.nFFT) + '.pth').cuda()
        Pf = weights_per_freq * weights_per_freq
        Pf_sum = Pf.sum()
        Pf = Pf / Pf_sum  # normalize
        Eval = lambda Wreal, Wimag, Hreal, Himag, Tlist: srcIndepSDR_freqpower(Wreal, Wimag, Hreal, Himag, Tlist, Pf)
    elif(args.eval_type == 'srcIndepSDR_mag_diffperT'):
        Eval = srcIndepSDR_mag_diffperT
    elif(args.eval_type == 'srcIndepSDR_Cproj_by_WH'):
        Eval = srcIndepSDR_Cproj_by_WH
    elif(args.eval_type == 'srcIndepSDR_Cproj_by_SShat'):
        #Eval = srcIndepSDR_Cproj_by_SShat
        print('srcIndepSDR_Cproj_by_SShat, eps = ' + str(args.eps))
        Eval = lambda clean_real, clean_imag, out_real, out_imag, Tlist: srcIndepSDR_Cproj_by_SShat(clean_real, clean_imag, out_real, out_imag, Tlist, args.eps)
    elif (args.eval_type == 'srcIndepSDR_freqpower_diffperT'):
        weights_per_freq = torch.load('weights_per_freq_' + str(args.nFFT) + '.pth').cuda()
        Pf = weights_per_freq * weights_per_freq
        Pf_sum = Pf.sum()
        Pf = Pf / Pf_sum  # normalize
        Eval = lambda Wreal, Wimag, Hreal, Himag, Tlist: srcIndepSDR_freqpower_diffperT(Wreal, Wimag, Hreal, Himag, Tlist, Pf)
    elif(args.eval_type == 'srcIndepSDR_freqpower_by_enhanced'):
        weights_per_freq = torch.load('weights_per_freq_' + str(args.nFFT) + '.pth').cuda()
        Pf = weights_per_freq * weights_per_freq
        Pf_sum = Pf.sum()
        Pf = Pf / Pf_sum  # normalize

        Eval = lambda out_real, out_imag, Tlist: srcIndepSDR_freqpower_by_enhanced(out_real, out_imag, Tlist, Pf)
    else:
        Eval = None

    if(args.eval2_type == 'sInvSDR_time'):
        Eval2 = sInvSDR_time
    elif(args.eval2_type == 'sInvSDR_spec'):
        Eval2 = sInvSDR_spec
    elif(args.eval2_type == 'sInvSDR_mag'):
        Eval2 = sInvSDR_mag
    elif(args.eval2_type == 'srcIndepSDR_mag'):
        Eval2 = srcIndepSDR_mag
    elif(args.eval2_type == 'srcIndepSDR_Cproj_by_WH'):
        Eval2 = srcIndepSDR_Cproj_by_WH
    elif(args.eval2_type == 'srcIndepSDR_Cproj_by_SShat'):
        #Eval2 = srcIndepSDR_Cproj_by_SShat
        print('srcIndepSDR_Cproj_by_SShat, eps = ' + str(args.eps))
        Eval2 = lambda clean_real, clean_imag, out_real, out_imag, Tlist: srcIndepSDR_Cproj_by_SShat(clean_real, clean_imag, out_real, out_imag, Tlist, args.eps)
    elif (args.eval2_type == 'srcIndepSDR_freqpower'):
        weights_per_freq = torch.load('weights_per_freq_' + str(args.nFFT) + '.pth').cuda()
        Pf = weights_per_freq * weights_per_freq
        Pf_sum = Pf.sum()
        Pf = Pf / Pf_sum  # normalize
        Eval2 = lambda Wreal, Wimag, Hreal, Himag, Tlist: srcIndepSDR_freqpower(Wreal, Wimag, Hreal, Himag, Tlist, Pf)
    elif(args.eval2_type == 'srcIndepSDR_mag_diffperT'):
        Eval2 = srcIndepSDR_mag_diffperT
    elif (args.eval2_type == 'srcIndepSDR_freqpower_diffperT'):
        weights_per_freq = torch.load('weights_per_freq_' + str(args.nFFT) + '.pth').cuda()
        Pf = weights_per_freq * weights_per_freq
        Pf_sum = Pf.sum()
        Pf = Pf / Pf_sum  # normalize
        Eval2 = lambda Wreal, Wimag, Hreal, Himag, Tlist: srcIndepSDR_freqpower_diffperT(Wreal, Wimag, Hreal, Himag, Tlist, Pf)
    elif(args.eval2_type == 'srcIndepSDR_freqpower_by_enhanced'):
        weights_per_freq = torch.load('weights_per_freq_' + str(args.nFFT) + '.pth').cuda()
        Pf = weights_per_freq * weights_per_freq
        Pf_sum = Pf.sum()
        Pf = Pf / Pf_sum  # normalize

        Eval2 = lambda out_real, out_imag, Tlist: srcIndepSDR_freqpower_by_enhanced(out_real, out_imag, Tlist, Pf)
    else:
        Eval2 = None

    # Network
    #utils.CPUmemDebug('before network init', mem_debug_file)
    if(args.model_type == 'unet'):
        from models.unet import Unet
        json_path = os.path.join(args.model_json)
        params = utils.Params(json_path)
        net = Unet(params.model, loss_type=args.loss_type, nMic = args.nMic, reverb_frame=args.reverb_frame,
                   use_depthwise=args.use_depthwise, nFreq = int((n_fft/args.ds_rate)/2+1), use_bn=args.use_BN,
                   input_type=args.input_type, ds_rate = args.ds_rate,
                   inverse_type = args.inverse_type, f_start_ratio=f_start_ratio, f_end_ratio=f_end_ratio,
                   ec_decomposition=args.ec_decomposition, carrier_input_indep=args.carrier_input_indep, ec_bias=args.ec_bias,
                   carrier_scale=args.carrier_init_scale)
        stride_product_time = get_stride_product_time(params.model['encoders'])
    elif(args.model_type == 'lcn'):
        from models.lcn import LCN
        net = LCN(nMic=args.nMic, nFreq=int((n_fft/args.ds_rate)/2+1), nHidden=args.nHidden, ksz_time=args.ksz_time, nLayer=args.nLayer,
                  use_bn=args.use_BN, input_type=args.input_type, ds_rate=args.ds_rate, reverb_frame=args.reverb_frame, CW_freq=args.CW_freq,
                  inverse_type = args.inverse_type)
        stride_product_time = 0

    elif(args.model_type == 'realunet'):
        from models.unet import RealUnet
        json_path = os.path.join(args.model_json)
        params = utils.Params(json_path)
        net = RealUnet(params.model, loss_type=args.loss_type, nMic = args.nMic, use_bn=args.use_BN).cuda()
        stride_product_time = get_stride_product_time(params.model['encoders'])
    #utils.CPUmemDebug('after network init', mem_debug_file)

    if(args.mode == 'train'):
        if(args.start_epoch > 0):
            print('training starts from epoch '+ str(args.start_epoch))

            checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar',
                                    map_location=lambda storage, loc: storage)

            print('load saved model') # order of netdefine-netload-netcuda-optimdefine-optimload-optimcuda is important
            net.load_state_dict(checkpoint['model'])
            net.cuda()

            # Optimizer
            optimizer = optim.Adam(net.parameters(), lr=args.lR0, amsgrad=True)

            print('load saved optimizer and move to gpu')
            optimizer.load_state_dict(checkpoint['optimizer']) # order is important
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            del checkpoint
            torch.cuda.empty_cache()
            logger = open('logs/log_' + str(args.expnum) + '.txt', 'a')
        else:
            net.cuda() # order of netdefine-netload-netcuda-optimdefine-optimload-optimcuda is important
            optimizer = optim.Adam(net.parameters(), lr=args.lR0, amsgrad=True)

            if(args.do_log):
                logger = open('logs/log_' + str(args.expnum) + '.txt', 'w')
            else:
                logger = None

        if not os.path.exists('wavs/' + str(args.expnum)):
            os.makedirs('wavs/' + str(args.expnum))

        # measure reverberant speech loss
        if (args.measure_reverb_loss):
            print('@epoch0: measuring reverberant training loss')
            rev_tr = measure_reverb_loss(train_loader, Loss, args.loss_type)
            print('rev_tr = ' + str(rev_tr))
            logger.write('rev_tr = ' + str(rev_tr) + '\n')

            print('@epoch0: measuring reverberant dt loss')
            rev_dt = measure_reverb_loss(val_loader, Loss, args.loss_type)
            print('rev_dt = ' + str(rev_dt))
            logger.write('rev_dt = ' + str(rev_dt) + '\n')

        # count parameter size
        nParam = count_parameters(net)
        print('#param = ' + str(nParam))

        # Learning rate scheduler
        #scheduler = ExponentialLR(optimizer, 0.95) # do not use scheduler anymore

        count = 0
        count_mb = 0
        count_eval = 0

        for epoch in range(args.start_epoch, args.num_epochs):
            # train
            loss_mb = 0
            eval_metric_mb = 0
            eval2_metric_mb = 0

            for _, input in enumerate(tqdm(train_loader)):
                count += 1
                count_mb += 1

                #utils.CPUmemDebug('before forward_common', mem_debug_file)
                if(not count % args.log_iter == 0):
                    loss, eval_metric, eval2_metric = \
                        forward_common(input, net, Loss, 'train', args.loss_type, args.eval_type, args.eval2_type,
                                                                     stride_product_time, mode='train', Eval=Eval, Eval2=Eval2,
                                       fix_len_by_cl=args.fix_len_by_cl, use_pos=args.ec_decomposition, eps=args.eps)
                    loss_mean = torch.mean(loss)
                    if(torch.isnan(loss_mean).item()):
                        print('NaN is detected on loss, terminate program')
                        logger.write('NaN is detected on loss, terminate program' + '\n')
                        sys.exit()
                    loss_mb += loss_mean.item()
                    eval_metric_mean = torch.mean(eval_metric).item()
                    eval_metric_mb += float(eval_metric_mean)
                    if(eval2_metric is not None):
                        eval2_metric_mean = torch.mean(eval2_metric).item()
                        eval2_metric_mb += float(eval2_metric_mean)
                else:
                    loss, eval_metric, eval2_metric = \
                        forward_common(input, net, Loss, 'train', args.loss_type, args.eval_type, args.eval2_type,
                                       stride_product_time, mode='train', expnum=args.expnum, Eval=Eval, Eval2=Eval2,
                                       fix_len_by_cl=args.fix_len_by_cl, use_pos=args.ec_decomposition, eps=args.eps)
                    loss_mean = torch.mean(loss)
                    if(torch.isnan(loss_mean).item()):
                        print('NaN is detected on loss, terminate program')
                        logger.write('NaN is detected on loss, terminate program' + '\n')
                        sys.exit()
                    eval_metric_mean = torch.mean(eval_metric).item()
                    loss_mb += loss_mean.item()
                    eval_metric_mb += float(eval_metric_mean)
                    if(eval2_metric is not None):
                        eval2_metric_mean = torch.mean(eval2_metric).item()
                        eval2_metric_mb += float(eval2_metric_mean)
                    print('train, epoch: ' + str(epoch) + ', loss: ' + str(loss_mean.item()))
                    print('train, epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_mean))
                    if(logger is not None):
                        logger.write('train, epoch: ' + str(epoch) + ', loss: ' + str(loss_mean.item()) + '\n')
                        logger.write('train, epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_mean) + '\n')
                    if(eval2_metric is not None):
                        print('train, epoch: ' + str(epoch) + ', eval2_metric: ' + str(eval2_metric_mean))
                        if(logger is not None):
                            logger.write('train, epoch: ' + str(epoch) + ', eval2_metric: ' + str(eval2_metric_mean) + '\n')

                    loss_mb = loss_mb/count_mb
                    eval_metric_mb = eval_metric_mb/count_mb
                    eval2_metric_mb = eval2_metric_mb/count_mb
                    #loss_w_var_mb = loss_w_var_mb/count_mb
                    print('train, epoch: ' + str(epoch) + ', loss (minibatch): ' + str(loss_mb))
                    print('train, epoch: ' + str(epoch) + ', eval_metric (minibatch): ' + str(eval_metric_mb))
                    print('train, epoch: ' + str(epoch) + ', eval2_metric (minibatch): ' + str(eval2_metric_mb))
                    #print('train, epoch: ' + str(epoch) + ', w_var (minibatch): ' + str(loss_w_var_mb))

                    if(logger is not None):
                        logger.write('train, epoch: ' + str(epoch) + ', loss: (minibatch): ' + str(loss_mb) + '\n')
                        logger.write('train, epoch: ' + str(epoch) + ', eval_metric: (minibatch): ' + str(eval_metric_mb) + '\n')
                        logger.write('train, epoch: ' + str(epoch) + ', eval2_metric: (minibatch): ' + str(eval2_metric_mb) + '\n')
                        #logger.write('train, epoch: ' + str(epoch) + ', w_var: (minibatch): ' + str(loss_w_var_mb) + '\n')
                        logger.flush()

                    loss_mb = 0
                    eval_metric_mb = 0
                    eval2_metric_mb = 0
                    count_mb = 0


                #utils.CPUmemDebug('before backward & step', mem_debug_file)

                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()
                #utils.CPUmemDebug('after backward & step', mem_debug_file)
                if(count % args.eval_iter == 0):
                    with torch.no_grad():
                        if(args.do_eval):
                            net.eval()

                        # Validaion
                        #utils.CPUmemDebug('before eval (val)', mem_debug_file)
                        count_eval += 1
                        evaluate(val_loader, net, Loss, 'val', args.loss_type, args.eval_type, args.eval2_type,
                                 stride_product_time, logger, epoch, Eval, Eval2,
                                 args.fix_len_by_cl, ec_decomposition=args.ec_decomposition, eps=args.eps)
                        #utils.CPUmemDebug('after eval (val)', mem_debug_file)
                        # Test
                        if (len(args.te1_manifest) > 0):
                            evaluate(test1_loader, net, Loss, 'test', args.loss_type, args.eval_type, args.eval2_type,
                                     stride_product_time, logger, epoch, Eval, Eval2,
                                     args.fix_len_by_cl, ec_decomposition=args.ec_decomposition, eps=args.eps)

                        # Test2
                        if (len(args.te2_manifest) > 0):
                            evaluate(test2_loader, net, Loss, 'test2', args.loss_type, args.eval_type, args.eval2_type,
                                     stride_product_time, logger, epoch, Eval, Eval2,
                                     args.fix_len_by_cl, ec_decomposition=args.ec_decomposition, eps=args.eps)

                        net.train()
                        gc.collect()
                        #print('!!!!!!!!!!!!! CURRENTLY, no gc.collect !!!!!!!!!!!!!!!!!!!!')
                        utils.CPUmemDebug('memory after gc.collect()', logger)
                        #trace_print(logger)

                    #utils.CPUmemDebug('before save NN', mem_debug_file)
                    torch.save({'epoch': epoch+1, 'model':net.state_dict(), 'optimizer': optimizer.state_dict()},
                               'checkpoint/' + str(args.expnum) + '-model.pth.tar')
                    #utils.CPUmemDebug('after save NN', mem_debug_file)
            torch.save({'epoch': epoch + 1, 'model': net.state_dict(), 'optimizer': optimizer.state_dict()},
                       'checkpoint/' + str(args.expnum) + '-model.pth.tar')
            #scheduler.step()
            torch.cuda.empty_cache()
        logger.close()
    elif(args.mode == 'test'):
        assert(0), 'not implemented yet'
    elif(args.mode == 'RT_analysis'):
        print('load pretrained model & optimizer')
        #checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar', map_location='cuda:' + str(args.gpu))
        checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar', map_location=lambda storage, loc: storage)

        print('load saved model')  # order of netdefine-netload-netcuda-optimdefine-optimload-optimcuda is important
        net.load_state_dict(checkpoint['model'])
        net.cuda()

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=args.lR0, amsgrad=True)

        print('load saved optimizer and move to gpu')
        optimizer.load_state_dict(checkpoint['optimizer'])  # order is important
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        RT_dir = 'RT_analysis/' + str(args.expnum)
        if not os.path.exists(RT_dir):
            os.makedirs(RT_dir)

        nTrain = 7861
        nValid = 1484

        # ver 1. type = torch.zeros --> cannot store path (=string)
        #SDR_RT_per_sample_tr = torch.zeros(nTrain, 3) # column: path, SDR, RT
        #SDR_RT_per_sample_dt = torch.zeros(nValid, 3) # column: path, SDR, RT

        # ver 2. separately save everythings
        reverb_paths_tr = []
        reverb_paths_dt = []
        SDR_RT_per_sample_tr = torch.zeros(nTrain, 2)  # column: SDR, RT
        SDR_RT_per_sample_dt = torch.zeros(nValid, 2)  # column: SDR, RT

        with torch.no_grad():
            # tr
            count = 0
            print('measuring training data performance')
            #data_bar = tqdm(train_loader)
            #for input in data_bar:
            for _, input in enumerate(tqdm(train_loader)):
            #for _, input in tqdm(enumerate(train_loader)):
            #for _, input in enumerate(train_loader):
                reverb_paths = input[6]
                RT60s = input[7]
                loss, eval_metric, eval2_metric = \
                    forward_common(input, net, Loss, 'tr', args.loss_type, args.eval_type, args.eval2_type,
                                   stride_product_time, expnum=args.expnum, mode='train', Eval=Eval, Eval2=Eval2, fix_len_by_cl=args.fix_len_by_cl,
                                   count=count, use_pos=args.ec_decomposition, eps=args.eps)
                N = eval_metric.size(0)

                for n in range(N):
                    reverb_paths_tr.append(reverb_paths[n])
                    SDR_RT_per_sample_tr[count*N + n, 0] = eval_metric[n].item()
                    SDR_RT_per_sample_tr[count * N + n, 1] = RT60s[n]

                count += 1

                #torch.save(RT_dir + '/performance_tr.py', SDR_RT_per_sample_tr)
                #np.save(RT_dir + '/reverb_path_tr.npy', reverb_paths_tr)

                sio.savemat(RT_dir + '/SDR_per_RT_tr.mat', {'SDR':SDR_RT_per_sample_tr.numpy()})

            # dt
            count = 0
            print('measuring validation data performance')
            #data_bar = tqdm(val_loader)
            #for input in data_bar:
            for _, input in enumerate(tqdm(val_loader)):
            #for _, input in tqdm(enumerate(val_loader)):
            #for _, input in enumerate(val_loader):
                reverb_paths = input[6]
                RT60s = input[7]
                loss, eval_metric, eval2_metric = forward_common(input, net, Loss, 'dt', args.loss_type, args.eval_type, args.eval2_type,
                                                    stride_product_time, expnum=args.expnum, mode='train',
                                                    Eval=Eval, Eval2=Eval2, fix_len_by_cl=args.fix_len_by_cl, count=count,
                                                    use_pos=args.ec_decomposition, eps=args.eps)
                N = eval_metric.size(0)
                for n in range(N):
                    reverb_paths_dt.append(reverb_paths[n])
                    SDR_RT_per_sample_dt[count * N + n, 0] = eval_metric[n].item()
                    SDR_RT_per_sample_dt[count * N + n, 1] = RT60s[n]

                count += 1

            sio.savemat(RT_dir + '/SDR_per_RT_dt.mat', {'SDR': SDR_RT_per_sample_dt.numpy()})

            # save reverberant path
            sio.savemat(RT_dir + '/reverb_path_tr.mat', {'reverb_path': np.asarray(tuple(reverb_paths_tr))})
            sio.savemat(RT_dir + '/reverb_path_dt.mat', {'reverb_path': np.asarray(tuple(reverb_paths_dt))})


    elif(args.mode == 'generate'):
        print('load pretrained model')
        checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar', map_location=lambda storage, loc: storage)
        '''
        net_dict = net.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if k in net_dict}
        net_dict.update(checkpoint)
        '''

        print('load saved model')  # order of netdefine-netload-netcuda-optimdefine-optimload-optimcuda is important
        net.load_state_dict(checkpoint['model'])
        net.cuda()

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=args.lR0, amsgrad=True)

        print('load saved optimizer and move to gpu')
        optimizer.load_state_dict(checkpoint['optimizer'])  # order is important
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()


        specs_dir = 'specs/' + str(args.expnum)
        if not os.path.exists(specs_dir):
            os.makedirs(specs_dir)

        with torch.no_grad():
            # tr
            count = 0
            eval_metric_total = 0

            if (len(args.tr_manifest) > 0):
                for _, input in enumerate(tqdm(train_loader)):
                #for _, input in tqdm(enumerate(train_loader)):
                #for _, input in enumerate(train_loader):
                    loss, eval_metric, eval2_metric = forward_common(input, net, Loss, 'tr', args.loss_type, args.eval_type, args.eval2_type,
                                                                        stride_product_time, expnum=args.expnum, fixed_src=args.fixed_src, mode='generate',
                                                                        Eval=Eval, Eval2=Eval2, fix_len_by_cl=args.fix_len_by_cl, count=count, use_pos=args.ec_decomposition
                                                                        ,save_activation=args.save_activation, eps=args.eps)
                    reverb_paths = []
                    for n in range(input[0].size(0)):
                        reverb_paths.append(input[3][n])
                    count = count + 1
                    if (eval2_metric is None):
                        sio.savemat('specs/' + str(args.expnum) + '/SDR_tr_' + str(count) + '.mat',
                                    {'loss': loss.data.cpu().numpy(), 'eval': eval_metric.data.cpu().numpy(),
                                     'reverb_path': np.asarray(tuple(reverb_paths))}) # no Cmag to save
                    else:
                        sio.savemat('specs/' + str(args.expnum) + '/SDR_tr_' + str(count) + '.mat',
                                {'loss': loss.data.cpu().numpy(), 'eval': eval_metric.data.cpu().numpy(),
                                 'eval2': eval2_metric.data.cpu().numpy(),
                                 'reverb_path': np.asarray(tuple(reverb_paths))})  # no Cmag to save

                    eval_metric_total += eval_metric.mean().item()
                    if(count == args.nGenerate):
                        break
                eval_metric_total = eval_metric_total/count
                print('tr SDR = ' + str(eval_metric_total))
            else:
                print('NO TRAINING MANIFEST')

            # dt
            if(len(args.val_manifest) > 0):
                count = 0
                eval_metric_total = 0

                for _, input in enumerate(tqdm(val_loader)):
                #for _, input in tqdm(enumerate(val_loader)):
                #for _, input in enumerate(val_loader):
                    loss, eval_metric, eval2_metric = forward_common(input, net, Loss, 'dt', args.loss_type, args.eval_type, args.eval2_type,
                                                           stride_product_time, expnum=args.expnum, fixed_src=args.fixed_src, mode='generate',
                                                           Eval=Eval, Eval2=Eval2, fix_len_by_cl=args.fix_len_by_cl, count=count,
                                                           use_pos=args.ec_decomposition,save_activation=args.save_activation, eps=args.eps)
                    count = count + 1
                    reverb_paths = []
                    for n in range(input[0].size(0)):
                        reverb_paths.append(input[3][n])
                    if(eval2_metric is None):
                        sio.savemat('specs/' + str(args.expnum) + '/SDR_dt_' + str(count) + '.mat',
                                {'loss': loss.data.cpu().numpy(), 'eval': eval_metric.data.cpu().numpy(),
                                 'reverb_path': np.asarray(tuple(reverb_paths))})  # no Cmag to save
                    else:
                        sio.savemat('specs/' + str(args.expnum) + '/SDR_dt_' + str(count) + '.mat',
                                    {'loss': loss.data.cpu().numpy(), 'eval': eval_metric.data.cpu().numpy(),
                                     'eval2': eval2_metric.data.cpu().numpy(),
                                     'reverb_path': np.asarray(tuple(reverb_paths))})  # no Cmag to save
                    eval_metric_total += eval_metric.mean().item()
                    if(count == args.nGenerate):
                        break
                eval_metric_total = eval_metric_total/count
                print('dt SDR = ' + str(eval_metric_total))
            else:
                print('NO VALIDATION MANIFEST')


def evaluate(loader, net, Loss, data_type, loss_type, eval_type, eval2_type, stride_product,
             logger, epoch, Eval, Eval2, fix_len_by_cl, ec_decomposition=False, eps=1e-10):
    count = 0
    loss_total = 0
    #loss_w_var_total = 0
    eval_metric_total = 0
    eval2_metric_total = 0

    # data_bar = tqdm(loader)
    # for input in data_bar:
    with torch.no_grad():
        for _, input in enumerate(tqdm(loader)):
        #for _, input in tqdm(enumerate(loader)):
        #for _, input in enumerate(loader):
            count += 1
            loss, eval_metric, eval2_metric = forward_common(input, net, Loss, data_type, loss_type, eval_type, eval2_type,
                                                             stride_product, mode='train',
                                               Eval=Eval, Eval2=Eval2, fix_len_by_cl=fix_len_by_cl, use_pos=ec_decomposition, eps=eps)
            loss_mean = torch.mean(loss)
            loss_total += loss_mean.item()
            eval_metric_mean = torch.mean(eval_metric)
            eval_metric_total += eval_metric_mean.item()
            if(eval2_metric is not None):
                eval2_metric_mean = torch.mean(eval2_metric)
                eval2_metric_total += eval2_metric_mean.item()

    loss_total = loss_total / len(loader)
    eval_metric_total = eval_metric_total / len(loader)
    eval2_metric_total = eval2_metric_total / len(loader)
    #loss_w_var_total = loss_w_var_total / len(loader)

    #loss_decrease = prev_loss_total - loss_total
    #avg_loss_decrease = (avg_loss_decrease * (eval_count - 1) + loss_decrease) / eval_count

    print(data_type + ', epoch: ' + str(epoch) + ', loss: ' + str(loss_total))
    print(data_type + ', epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_total))
    print(data_type + ', epoch: ' + str(epoch) + ', eval2_metric: ' + str(eval2_metric_total))
    #print(data_type + ', epoch: ' + str(epoch) + ', w_var: ' + str(loss_w_var_total))
    #print(data_type + ', epoch: ' + str(epoch) + ', loss_decrease: ' + str(loss_decrease) + ', avg : ' + str(avg_loss_decrease))
    if(logger is not None):
        logger.write(data_type + ', epoch: ' + str(epoch) + ', loss: ' + str(loss_total) + '\n')
        logger.write(data_type + ', epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_total) + '\n')
        logger.write(data_type + ', epoch: ' + str(epoch) + ', eval2_metric: ' + str(eval2_metric_total) + '\n')
        #logger.write(data_type + ', epoch: ' + str(epoch) + ', w_var: ' + str(loss_w_var_total) + '\n')
        #logger.write(data_type + ', epoch: ' + str(epoch) + ', loss_decrease: ' + str(loss_decrease) + ', avg : ' + str(avg_loss_decrease) + '\n')

        logger.flush()

    #return loss_total, avg_loss_decrease # no need to return

if __name__ == '__main__':
    config, unparsed = get_config()

    main(config)
