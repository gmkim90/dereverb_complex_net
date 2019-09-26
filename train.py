import os
import sys
import torch
import numpy as np

import torch.optim as optim

from tqdm import tqdm
import gc

import utils
from utils import get_stride_product_time, count_parameters

import models.loss as losses

import pickle
from se_dataset import SpecDataset
from torch.utils.data import DataLoader
import scipy.io as sio
#from memprof import *

import pdb

from essential import forward_common
from config import get_config

def main(args):
    #torch.autograd.set_detect_anomaly(True)
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
            if(args.fix_src):
                if(len(args.tr_manifest) == 0):
                    args.tr_manifest = prefix + 'L553_30cm_30cm_' + str(args.grid_cm) + 'cm_nSrc_' + str(args.fix_src_n) + '.csv'

                # comparison purpose
                #if(len(args.val_manifest) == 0):
                    #args.val_manifest = prefix + 'L553_fixedmic_' + args.directivity + '_' + str(args.grid_cm) + 'cm' + '_RT' + str(args.RT) + '_val.csv'
            else:
                if(args.grid_cm == 1):
                    # new data (large room, fixed mic)
                    #pdb.set_trace()
                    if(len(args.tr_manifest) == 0):
                        args.tr_manifest = prefix + 'L553_fixedmic_' + args.directivity + '_' + str(args.grid_cm) + 'cm' + '_RT' + str(args.RT) + '_tr.csv'

                    if(args.save_wav and len(args.trsub_manifest) == 0):
                        args.trsub_manifest = prefix + 'L553_fixedmic_' + args.directivity + '_' + str(args.grid_cm) + 'cm' + '_RT' + str(args.RT) + '_trsub_8samples.csv'

                    if (len(args.val_manifest) == 0):
                        args.val_manifest = prefix + 'L553_fixedmic_' + args.directivity + '_' + str(args.grid_cm) + 'cm' + '_RT' + str(args.RT) + '_val.csv'

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

    if(args.save_wav):
        savename_ISTFT = 'ISTFT_' + str(n_fft) + '.pth'
        from models.layers.istft import ISTFT
        if not os.path.exists(savename_ISTFT):
            print('init ISTFT')
            istft = ISTFT(n_fft, hop_length, window='hanning')
            with open(savename_ISTFT, 'wb') as f:
                pickle.dump(istft, f)
        else:
            print('load saved ISTFT')
            with open(savename_ISTFT, 'rb') as f:
                istft = pickle.load(f)
        istft = istft.cuda()
    else:
        istft = None

    #print('2')
    f_start_ratio = args.f_start / (args.fs / 2)
    f_end_ratio = args.f_end / (args.fs / 2)

    #loss_w_var = 0 #dummy variable

    stft = lambda x: torch.stft(x, n_fft, hop_length, win_length = win_size, window=window)

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
                                    load_IR=args.load_IR, use_localization=args.use_localization, src_range=src_range_list,
                                    nSource=args.nSource,
                                    start_ratio=args.start_ratio, end_ratio=args.end_ratio,
                                    clamp_frame=args.clamp_frame, ref_mic_direct_td_subtract=args.ref_mic_direct_td_subtract,
                                    interval_cm=args.interval_cm_tr, use_audio=args.save_wav)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate, shuffle=shuffle_train_loader, num_workers=0)

    if (len(args.trsub_manifest) > 0):
        trsub_dataset = SpecDataset(manifest_path=args.trsub_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl, return_path=args.return_path,
                                    load_IR=args.load_IR, use_localization=args.use_localization, src_range=src_range_list,
                                    nSource=args.nSource,
                                    start_ratio=args.start_ratio, end_ratio=args.end_ratio,
                                    clamp_frame=args.clamp_frame, ref_mic_direct_td_subtract=args.ref_mic_direct_td_subtract,
                                    interval_cm=args.interval_cm_tr, use_audio=args.save_wav)
        trsub_loader = DataLoader(dataset=trsub_dataset, batch_size=args.batch_size, collate_fn=trsub_dataset.collate, shuffle=False, num_workers=0)



    if (len(args.val_manifest) > 0):
        val_dataset = SpecDataset(manifest_path=args.val_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl, return_path=args.return_path,
                                  load_IR=args.load_IR, use_localization=args.use_localization, src_range='all',
                                  nSource=args.nSource,
                                  clamp_frame=args.clamp_frame, ref_mic_direct_td_subtract=args.ref_mic_direct_td_subtract,
                                  interval_cm=args.interval_cm_te, use_audio=args.save_wav)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate, shuffle=False, num_workers=0)

    if(len(args.te1_manifest) > 0):
        test1_dataset = SpecDataset(manifest_path=args.te1_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl,
                                    load_IR=args.load_IR, use_localization=args.use_localization, src_range='all',
                                    nSource=args.nSource,
                                    clamp_frame=args.clamp_frame, ref_mic_direct_td_subtract=args.ref_mic_direct_td_subtract,
                                    interval_cm=args.interval_cm_te, use_audio=args.save_wav)
        test1_loader = DataLoader(dataset=test1_dataset, batch_size=args.batch_size, collate_fn=test1_dataset.collate, shuffle=False, num_workers=0)

    if(len(args.te2_manifest) > 0):
        test2_dataset = SpecDataset(manifest_path=args.te2_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl,
                                    load_IR=args.load_IR, use_localization=args.use_localization, src_range='all',
                                    nSource=args.nSource,
                                    clamp_frame=args.clamp_frame, ref_mic_direct_td_subtract=args.ref_mic_direct_td_subtract,
                                    interval_cm=args.interval_cm_te, use_audio=args.save_wav) # for test2, set pos_range as 'all' (all positions within a room)
        test2_loader = DataLoader(dataset=test2_dataset, batch_size=args.batch_size, collate_fn=test2_dataset.collate, shuffle=False, num_workers=0)

    torch.set_printoptions(precision=10, profile="full")
    #utils.CPUmemDebug('after dataset init', mem_debug_file)

    # Set loss type
    if(len(args.loss_type) > 0):
        Loss = getattr(losses, args.loss_type)
    else:
        Loss = None

    if(len(args.eval_type) > 0):
        Eval = getattr(losses, args.eval_type)
    else:
        Eval = None

    if (len(args.eval2_type) > 0):
        Eval2 = getattr(losses, args.eval2_type)
    else:
        Eval2 = None

    # Network
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
        if (args.eval_iter == 0 or args.eval_iter > len(train_loader)):
            args.eval_iter = len(train_loader)

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
                                       fix_len_by_cl=args.fix_len_by_cl, save_wav=args.save_wav, istft=istft)
                    loss_mean = torch.mean(loss)
                    if(torch.isnan(loss_mean).item()):
                        print('NaN is detected on loss, terminate program')
                        logger.write('NaN is detected on loss, terminate program' + '\n')
                        sys.exit()
                    loss_mb += loss_mean.item()
                    if(eval_metric is not None):
                        eval_metric_mean = torch.mean(eval_metric).item()
                        eval_metric_mb += float(eval_metric_mean)
                    if(eval2_metric is not None):
                        eval2_metric_mean = torch.mean(eval2_metric).item()
                        eval2_metric_mb += float(eval2_metric_mean)
                else:
                    loss, eval_metric, eval2_metric = \
                        forward_common(input, net, Loss, 'train', args.loss_type, args.eval_type, args.eval2_type,
                                       stride_product_time, mode='train', expnum=args.expnum, Eval=Eval, Eval2=Eval2,
                                       fix_len_by_cl=args.fix_len_by_cl, save_wav=args.save_wav, istft=istft)
                    loss_mean = torch.mean(loss)
                    if(torch.isnan(loss_mean).item()):
                        print('NaN is detected on loss, terminate program')
                        logger.write('NaN is detected on loss, terminate program' + '\n')
                        sys.exit()
                    loss_mb += loss_mean.item()
                    if(eval_metric is not None):
                        eval_metric_mean = torch.mean(eval_metric).item()
                        eval_metric_mb += float(eval_metric_mean)
                    if(eval2_metric is not None):
                        eval2_metric_mean = torch.mean(eval2_metric).item()
                        eval2_metric_mb += float(eval2_metric_mean)
                    print('train, epoch: ' + str(epoch) + ', loss: ' + str(loss_mean.item()))
                    logger.write('train, epoch: ' + str(epoch) + ', loss: ' + str(loss_mean.item()) + '\n')

                    if (eval_metric is not None):
                        print('train, epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_mean))
                        logger.write('train, epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_mean) + '\n')

                    if(eval2_metric is not None):
                        print('train, epoch: ' + str(epoch) + ', eval2_metric: ' + str(eval2_metric_mean))
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
                        count_eval += 1

                        # Training subset
                        if (len(args.trsub_manifest) > 0):
                            evaluate(args.expnum, trsub_loader, net, Loss, 'trsub', args.loss_type, args.eval_type, args.eval2_type,
                                     stride_product_time, logger, epoch, Eval, Eval2,
                                     args.fix_len_by_cl, save_wav=args.save_wav, istft=istft)

                        # Validaion
                        if (len(args.te1_manifest) > 0):
                            evaluate(args.expnum, val_loader, net, Loss, 'val', args.loss_type, args.eval_type, args.eval2_type,
                                     stride_product_time, logger, epoch, Eval, Eval2,
                                     args.fix_len_by_cl, save_wav=args.save_wav, istft=istft)
                        #utils.CPUmemDebug('after eval (val)', mem_debug_file)
                        # Test
                        if (len(args.te1_manifest) > 0):
                            evaluate(args.expnum, test1_loader, net, Loss, 'test', args.loss_type, args.eval_type, args.eval2_type,
                                     stride_product_time, logger, epoch, Eval, Eval2,
                                     args.fix_len_by_cl, save_wav=args.save_wav, istft=istft)

                        # Test2
                        if (len(args.te2_manifest) > 0):
                            evaluate(args.expnum, test2_loader, net, Loss, 'test2', args.loss_type, args.eval_type, args.eval2_type,
                                     stride_product_time, logger, epoch, Eval, Eval2,
                                     args.fix_len_by_cl,save_wav=args.save_wav, istft=istft)

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
                    count = count + 1
                    if(args.skip_if_gen_exists and os.path.exists('specs/' + str(args.expnum) + '/SDR_tr_' + str(count+1) + '.mat')):
                        print('skip generating ' + 'specs/' + str(args.expnum) + '/SDR_tr_' + str(count) + '.mat')
                    else:
                        loss, eval_metric, eval2_metric = forward_common(input, net, Loss, 'tr', args.loss_type, args.eval_type, args.eval2_type,
                                                                         stride_product_time, expnum=args.expnum, fixed_src=args.fixed_src, mode='generate',
                                                                            Eval=Eval, Eval2=Eval2, fix_len_by_cl=args.fix_len_by_cl, count=count,
                                                                         save_activation=args.save_activation)
                        reverb_paths = []
                        for n in range(input[0].size(0)):
                            reverb_paths.append(input[3][n])

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
                    loss, eval_metric, eval2_metric = forward_common(input, net, Loss, 'dt', args.loss_type, args.eval_type, args.eval2_type,
                                                           stride_product_time, expnum=args.expnum, fixed_src=args.fixed_src, mode='generate',
                                                           Eval=Eval, Eval2=Eval2, fix_len_by_cl=args.fix_len_by_cl, count=count,
                                                           save_activation=args.save_activation)
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


def evaluate(expnum, loader, net, Loss, data_type, loss_type, eval_type, eval2_type, stride_product,
             logger, epoch, Eval, Eval2, fix_len_by_cl, save_wav=False, istft=None):
    count = 0
    loss_total = 0
    #loss_w_var_total = 0
    eval_metric_total = 0
    eval2_metric_total = 0

    # data_bar = tqdm(loader)
    # for input in data_bar:
    with torch.no_grad():
        for _, input in enumerate(tqdm(loader)):
            count += 1
            loss, eval_metric, eval2_metric = forward_common(input, net, Loss, data_type, loss_type, eval_type, eval2_type,
                                                             stride_product, mode='train', expnum=expnum,
                                               Eval=Eval, Eval2=Eval2, fix_len_by_cl=fix_len_by_cl,
                                                             save_wav=save_wav, istft=istft)
            save_wav = False # MAKE save_wav activate only once

            loss_mean = torch.mean(loss)
            loss_total += loss_mean.item()
            if(eval_metric is not None):
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
