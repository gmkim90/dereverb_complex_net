import os
import torch
import numpy as np
import sys

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm

import utils
from utils import get_stride_product_time, count_parameters
from models.layers.istft import ISTFT
from models.loss import cossim_time, cossim_spec, cossim_mag, sInvSDR_time, sInvSDR_spec, negative_MSE, sInvSDR_mag
from se_dataset import SpecDataset
from torch.utils.data import DataLoader
import scipy.io as sio


from essential import forward_common, eval_segment
from config import get_config

import gc

import time

import pdb

# n_fft, hop_length = 400, 160
#n_fft, hop_length = 512, 256

def main(args):
    assert (args.expnum >= 1)
    #assert (args.gpu >= 0)
    if (args.nMic == 8):
        assert (args.mic_sampling == 'no')
    if (args.mic_sampling == 'no'):
        assert (args.nMic == 8)
    if (args.mic_sampling == 'ref_manual'):
        args.subset1 = args.subset1.split(',')  # string to list
        args.subset2 = args.subset2.split(',')  # string to list

    prefix = 'data_sorted/'

    #args.val_trainIR_manifest = prefix + 'reverb_dt_simu_8ch_trainIR_paired.csv'

    #if(args.REV_VIS):
    if(args.dataset == 'DSIR' and args.grid_cm > 0):
        args.tr_manifest = prefix + 'L221_10105_grid' + str(args.grid_cm) + 'cm' + '_tr.csv'
        args.val_manifest = prefix + 'L221_10105_grid' + str(args.grid_cm) + 'cm' + '_val.csv'
        #args.te_smallset_manifest = prefix + 'L221_10105_grid' + str(args.grid_cm) + 'cm' + '_te.csv'
        args.te_smallset_manifest = ''
    elif(args.dataset == 'REV_VIS'):
        args.tr_manifest = prefix + 'RIV-VIS_tr.csv'
        #args.tr_manifest = prefix + 'visualize_scenario_augmented_tr.csv'
        #args.val_manifest = prefix + 'RIV-VIS_val.csv' # use original reverb dataset
        args.te_smallset_manifest = prefix + 'visualize_scenario_augmented_te_smallset.csv'
    elif(args.dataset == 'diverse_room'):
        if(args.mode == 'WER'):
            #args.tr_manifest = prefix + 'diverse_room_tr_txt.csv'
            args.tr_manifest = prefix + 'diverse_room_trsub_txt.csv' # use subset for evaluation purpose
            args.val_manifest = prefix + 'diverse_room_val_txt.csv'
        else:
            args.tr_manifest = prefix + 'diverse_room_tr.csv'
            args.val_manifest = prefix + 'diverse_room_val.csv'
        args.te_smallset_manifest = ''

    if(args.mode == 'RT_analysis'):
        args.tr_manifest = prefix + 'reverb_tr_simu_8ch_RT60.csv'
        args.val_manifest = prefix + 'reverb_dt_simu_8ch_RT60.csv'
        args.te_smallset_manifest = ''

    #return_time_delay = False
    #if(return_time_delay):
        #assert(not args.mic_sampling == 'random'), 'random mic sampling cannot use time delay for now'

    #if (args.cut_dtet):
#        args.val_cut_manifest = 'data_sorted/reverb_dt_simu_8ch_cut_paired.csv'
#        args.te_smallset_cut_manifest = 'data_sorted/reverb_et_hard6cases_simu_8ch_cut_paired.csv'

    #torch.cuda.set_device(args.gpu) # use CUDA_VISIBLE_DEVICES is more safe way

    n_fft = args.nFFT
    win_size = args.nWin
    if(args.hop_length == 0):
        hop_length = int(win_size/2)
    else:
        hop_length = args.hop_length
    window_path = 'window_' + str(win_size) + '.pth'
    window = torch.load(window_path)
    f_start_ratio = args.f_start / (args.fs / 2)
    f_end_ratio = args.f_end / (args.fs / 2)

    loss_w_var = 0 #dummy variable

    #if (args.fband_SDR):
        #interval_list = np.rint(np.linspace(0, int(n_fft/2+1), args.nFinterval)).astype(int)

    stft = lambda x: torch.stft(x, n_fft, hop_length, win_length = win_size, window=window)
    if(args.mode == 'WER'):
        import librosa
        AAS_path = '/data2/kenkim/AAS_enhancement/Speech_enhancement_by_AAS'
        sys.path.append(AAS_path)
        import model as AAS
        from decoder import GreedyDecoder
        import json

        win_size_AM = 320
        hop_len_AM = 160
        n_fft_AM = 320
        nMel = 40
        fs = 16000
        window_AM = torch.hamming_window(win_size_AM).cuda()

        stft_AM = lambda x: torch.stft(x, n_fft_AM, hop_len_AM, win_length = win_size_AM, window=window_AM)

        mel_basis = librosa.filters.mel(fs, n_fft_AM, nMel)
        stft_to_lmfb = AAS.STFT_to_LMFB(mel_basis, window_change=False)

        logger = open('logs/WER_' + str(args.expnum) + '.txt', 'w')

        with open(AAS_path + '/labels.json') as label_file:
            labels = str(''.join(json.load(label_file)))
        decoder = GreedyDecoder(labels)
    else:
        labels=None

    if(args.use_ISTFT or args.mode == 'WER'):
        istft = ISTFT(n_fft, hop_length, window='hanning').cuda()
    else:
        istft = None

    if(args.mode == 'generate'):
        args.return_path = True
        shuffle_train_loader = False
    else:
        shuffle_train_loader = True

    train_dataset = SpecDataset(manifest_path=args.tr_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl, return_path=args.return_path, labels=labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate, shuffle=shuffle_train_loader, num_workers=0)

    if (len(args.val_manifest) > 0):
        val_dataset = SpecDataset(manifest_path=args.val_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl, return_path=args.return_path, labels=labels)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate, shuffle=False, num_workers=0)

    if(args.eval_iter == 0):
        args.eval_iter = len(train_loader)

    if(len(args.te_smallset_manifest) > 0):
        test_dataset = SpecDataset(manifest_path=args.te_smallset_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl, labels=labels)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate, shuffle=False, num_workers=0)

    if(len(args.val_trainIR_manifest) > 0):
        val_trainIR_dataset = SpecDataset(manifest_path=args.val_trainIR_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl, labels=labels)
        val_trainIR_loader = DataLoader(dataset=val_trainIR_dataset, batch_size=args.batch_size, collate_fn = val_trainIR_dataset.collate, shuffle=False, num_workers=0)

    if (len(args.te_smallset_trainIR_manifest) > 0):
        test_trainIR_dataset = SpecDataset(manifest_path=args.te_smallset_trainIR_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl, labels=labels)
        test_trainIR_loader = DataLoader(dataset=test_trainIR_dataset, batch_size=args.batch_size, collate_fn = test_trainIR_dataset.collate, shuffle=False, num_workers=0)

    if(args.fband_SDR):
        trainsub_dataset = SpecDataset(manifest_path=args.trsub_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, fix_len_by_cl=args.fix_len_by_cl)
        trainsub_loader = DataLoader(dataset=trainsub_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate, shuffle=True, num_workers=0)

    torch.set_printoptions(precision=10, profile="full")

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
    elif(args.loss_type == 'negative_MSE'):
        Loss = negative_MSE

    if(args.eval_type == 'sInvSDR_time'):
        Eval = sInvSDR_time
    elif(args.eval_type == 'sInvSDR_spec'):
        Eval = sInvSDR_spec
    elif(args.eval_type == 'sInvSDR_mag'):
        Eval = sInvSDR_mag
    else:
        Eval = None

    # Network
    if(args.model_type == 'unet'):
        from models.unet import Unet
        json_path = os.path.join(args.model_json)
        params = utils.Params(json_path)
        net = Unet(params.model, loss_type=args.loss_type, nMic = args.nMic, reverb_frame=args.reverb_frame,
                   use_depthwise=args.use_depthwise, nFreq = int((n_fft/args.ds_rate)/2+1), use_bn=args.use_BN,
                   input_type=args.input_type, ds_rate = args.ds_rate,
                   inverse_type = args.inverse_type, f_start_ratio=f_start_ratio, f_end_ratio=f_end_ratio).cuda()
        stride_product_time = get_stride_product_time(params.model['encoders'])
    elif(args.model_type == 'lcn'):
        from models.lcn import LCN
        net = LCN(nMic=args.nMic, nFreq=int((n_fft/args.ds_rate)/2+1), nHidden=args.nHidden, ksz_time=args.ksz_time, nLayer=args.nLayer,
                  use_bn=args.use_BN, input_type=args.input_type, ds_rate=args.ds_rate, reverb_frame=args.reverb_frame, CW_freq=args.CW_freq,
                  inverse_type = args.inverse_type).cuda()
        stride_product_time = 0

    elif(args.model_type == 'realunet'):
        from models.unet import RealUnet
        json_path = os.path.join(args.model_json)
        params = utils.Params(json_path)
        net = RealUnet(params.model, loss_type=args.loss_type, nMic = args.nMic, use_bn=args.use_BN).cuda()
        stride_product_time = get_stride_product_time(params.model['encoders'])

    if(args.mode == 'train'):
        if(args.start_epoch > 0):
            print('training starts from epoch '+ str(args.start_epoch) )
            checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar',
                                    map_location=lambda storage, loc: storage)
            net.load_state_dict(checkpoint)
            del checkpoint
            logger = open('logs/log_' + str(args.expnum) + '.txt', 'a')
        else:
            logger = open('logs/log_' + str(args.expnum) + '.txt', 'w')

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

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=args.lR0, amsgrad=True)

        # Learning rate scheduler
        #scheduler = ExponentialLR(optimizer, 0.95) # do not use scheduler anymore

        count = 0
        count_mb = 0
        for epoch in range(args.start_epoch, args.num_epochs):
            # train
            loss_mb = 0
            loss_w_var_mb = 0
            eval_metric_mb = 0

            #train_bar = tqdm(train_loader)
            #for input in train_bar:
            for _, input in enumerate(tqdm(train_loader)):
                count += 1
                count_mb += 1

                # slice clean feature by frequency if needed --> move to forward_common
                '''
                if(args.f_end < args.fs):
                    cleanSTFT = input[1] # NxFxTx2
                    K = cleanSTFT.size(1)
                    k_start = int(K*args.f_start/(args.fs/2))
                    k_end = int(K * args.f_end / (args.fs / 2))
                    input[1] = cleanSTFT[:, k_start:k_end+1, :]
                '''
                if(not count % args.log_iter == 0):
                    loss, eval_metric = forward_common(istft, input, net, Loss, 'train', args.loss_type, stride_product_time, mode='train', Eval=Eval, fix_len_by_cl=args.fix_len_by_cl, w_var=args.w_var, f_start_ratio=f_start_ratio, f_end_ratio=f_end_ratio)
                    if(args.w_var == 0):
                        loss = torch.mean(loss)
                    elif(args.w_var > 0):
                        loss_w_var = loss[1]
                        loss = torch.mean(loss[0])
                        loss_w_var_mb += loss_w_var.item()
                    loss_mb += loss.item()
                    eval_metric = torch.mean(eval_metric).item()
                    eval_metric_mb += float(eval_metric)
                else:
                    loss, eval_metric = forward_common(istft, input, net, Loss, 'train', args.loss_type, stride_product_time, mode='train', save_wav=args.save_tr_wav, expnum=args.expnum, Eval=Eval, fix_len_by_cl=args.fix_len_by_cl, w_var=args.w_var, f_start_ratio=f_start_ratio, f_end_ratio=f_end_ratio)
                    if(args.w_var == 0):
                        loss = torch.mean(loss)
                    elif(args.w_var > 0):
                        loss_w_var = loss[1]
                        loss = torch.mean(loss[0])
                    eval_metric = torch.mean(eval_metric).item()
                    loss_mb += loss.item()
                    eval_metric_mb += float(eval_metric)
                    print('train, epoch: ' + str(epoch) + ', loss: ' + str(loss.item()))
                    logger.write('train, epoch: ' + str(epoch) + ', loss: ' + str(loss.item()) + '\n')
                    print('train, epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric))
                    logger.write('train, epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric) + '\n')

                    if(args.w_var > 0):
                        loss = loss + loss_w_var # coefficients are multiplied at forward_common()
                        loss_w_var = loss_w_var.item()
                        loss_w_var_mb += loss_w_var
                        print('train, epoch: ' + str(epoch) + ', w_var: ' + str(loss_w_var))
                        logger.write('train, epoch: ' + str(epoch) + ', w_var: ' + str(loss_w_var) + '\n')

                    loss_mb = loss_mb/count_mb
                    eval_metric_mb = eval_metric_mb/count_mb
                    loss_w_var_mb = loss_w_var_mb/count_mb
                    print('train, epoch: ' + str(epoch) + ', loss (minibatch): ' + str(loss_mb))
                    logger.write('train, epoch: ' + str(epoch) + ', loss: (minibatch): ' + str(loss_mb) + '\n')
                    print('train, epoch: ' + str(epoch) + ', eval_metric (minibatch): ' + str(eval_metric_mb))
                    logger.write('train, epoch: ' + str(epoch) + ', eval_metric: (minibatch): ' + str(eval_metric_mb) + '\n')
                    print('train, epoch: ' + str(epoch) + ', w_var (minibatch): ' + str(loss_w_var_mb))
                    logger.write('train, epoch: ' + str(epoch) + ', w_var: (minibatch): ' + str(loss_w_var_mb) + '\n')

                    loss_mb = 0
                    eval_metric_mb = 0
                    count_mb = 0
                    logger.flush()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if(count % args.eval_iter == 0):
                    with torch.no_grad():
                        if(args.do_eval):
                            net.eval()

                        # Validaion
                        evaluate(istft, val_loader, net, Loss, 'val', args.loss_type, stride_product_time, logger, epoch, Eval, args.fix_len_by_cl, args.w_var, f_start_ratio=f_start_ratio, f_end_ratio=f_end_ratio)

                        # Test
                        if (len(args.te_smallset_manifest) > 0):
                            evaluate(istft, test_loader, net, Loss, 'test', args.loss_type, stride_product_time, logger, epoch, Eval, args.fix_len_by_cl, args.w_var, f_start_ratio=f_start_ratio, f_end_ratio=f_end_ratio)

                        # Validation (trainIR) & Test (trainIR)
                        if (len(args.val_trainIR_manifest) > 0):
                            evaluate(istft, val_trainIR_loader, net, Loss, 'val_trainIR', args.loss_type, stride_product_time, logger, epoch, Eval, args.fix_len_by_cl, args.w_var)

                        # Test
                        if (len(args.te_smallset_trainIR_manifest) > 0):
                            evaluate(istft, test_trainIR_loader, net, Loss, 'test_trainIR', args.loss_type, stride_product_time, logger, epoch, Eval, args.fix_len_by_cl, args.w_var)

                        if(args.fband_SDR):
                            # training subset
                            evaluate_segment(interval_list, trainsub_loader, net, Eval, 'trainsub_segment', stride_product_time, logger, epoch, args.fix_len_by_cl)

                            # test (RT700 smallset)
                            evaluate_segment(interval_list, test_loader, net, Eval, 'test_segment', stride_product_time, logger, epoch, args.fix_len_by_cl)

                        net.train()
                        gc.collect()
                    torch.save(net.state_dict(), 'checkpoint/' + str(args.expnum) + '-model.pth.tar')
            #scheduler.step()
        logger.close()
    elif(args.mode == 'WER'):
        from essential import forward_WER

        print('load pretrained enhancement model')
        checkpoint_E = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar', map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint_E)
        net.eval()

        print('load pretrained acoustic model')
        AM_path = '/data2/kenkim/AAS_enhancement/AM_training/models/librispeech_25_final.pth.tar'
        package_AM = torch.load(AM_path, map_location=lambda storage, loc: storage)
        net_AM = AAS.DeepSpeech.load_model_package(package_AM)
        net_AM.cuda()
        net_AM.eval()

        WER_tr = utils.AverageMeter()
        WER_val = utils.AverageMeter()
        CER_tr = utils.AverageMeter()
        CER_val = utils.AverageMeter()
        Loss_tr = utils.AverageMeter()
        Loss_val = utils.AverageMeter()

        with torch.no_grad():
            # tr
            #data_bar = tqdm(train_loader)
            #for input in data_bar:
            for _, input in enumerate(tqdm(train_loader)):
                wer, cer, loss = forward_WER(istft, stft_AM, input, net, net_AM, stft_to_lmfb, Loss, args.decoding_type, stride_product_time, decoder, logger, fix_len_by_cl = args.fix_len_by_cl, expnum=args.expnum)
                N = input[0].size(0)
                WER_tr.update(wer, N)
                CER_tr.update(cer, N)
                Loss_tr.update(loss, N)
            print('train, loss ' + str(Loss_tr.avg))
            logger.write('train, loss ' + str(Loss_tr.avg) + '\n')
            print('train, wer ' + str(WER_tr.avg))
            logger.write('train, wer ' + str(WER_tr.avg) + '\n')
            print('train, cer ' + str(CER_tr.avg))
            logger.write('train, cer ' + str(CER_tr.avg) + '\n')

            # val
            #data_bar = tqdm(val_loader)
            #for input in data_bar:
            for _, input in enumerate(tqdm(val_loader)):
                wer, cer, loss = forward_WER(istft, stft_AM, input, net, net_AM, stft_to_lmfb, Loss, args.decoding_type, stride_product_time, decoder, logger, fix_len_by_cl = args.fix_len_by_cl, expnum=args.expnum)
                N = input[0].size(0)
                WER_val.update(wer, N)
                CER_val.update(cer, N)
                Loss_val.update(loss, N)

            print('valid, loss ' + str(Loss_val.avg))
            logger.write('valid, loss ' + str(Loss_val.avg) + '\n')
            print('valid, wer ' + str(WER_val.avg))
            logger.write('valid, wer ' + str(WER_val.avg) + '\n')
            print('valid, cer ' + str(CER_val.avg))
            logger.write('valid, cer ' + str(CER_val.avg) + '\n')

    elif(args.mode == 'RT_analysis'):
        print('load pretrained model')
        #checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar', map_location='cuda:' + str(args.gpu))
        checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar', map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint)
        net.eval()

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
                reverb_paths = input[6]
                RT60s = input[7]
                loss, eval_metric = forward_common(istft, input, net, Loss, 'tr', args.loss_type, stride_product_time, expnum=args.expnum, mode='train', Eval=Eval, fix_len_by_cl=args.fix_len_by_cl, count=count, w_var=args.w_var, f_start_ratio=f_start_ratio, f_end_ratio=f_end_ratio)
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
                reverb_paths = input[6]
                RT60s = input[7]
                loss, eval_metric = forward_common(istft, input, net, Loss, 'dt', args.loss_type,
                                                    stride_product_time, expnum=args.expnum, mode='train',
                                                    Eval=Eval, fix_len_by_cl=args.fix_len_by_cl, count=count,
                                                    w_var=args.w_var, f_start_ratio=f_start_ratio,
                                                    f_end_ratio=f_end_ratio)
                N = eval_metric.size(0)
                for n in range(N):
                    reverb_paths_dt.append(reverb_paths[n])
                    SDR_RT_per_sample_dt[count * N + n, 0] = eval_metric[n].item()
                    SDR_RT_per_sample_dt[count * N + n, 1] = RT60s[n]

                count += 1

            #torch.save(RT_dir + '/performance_dt.py', SDR_RT_per_sample_dt)
            #np.save(RT_dir + '/reverb_path_dt.npy', reverb_paths_dt)

            sio.savemat(RT_dir + '/SDR_per_RT_dt.mat', {'SDR': SDR_RT_per_sample_dt.numpy()})

            # save reverberant path
            sio.savemat(RT_dir + '/reverb_path_tr.mat', {'reverb_path': np.asarray(tuple(reverb_paths_tr))})
            sio.savemat(RT_dir + '/reverb_path_dt.mat', {'reverb_path': np.asarray(tuple(reverb_paths_dt))})


    elif(args.mode == 'generate'):
        print('load pretrained model')
        #checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar', map_location='cuda:' + str(args.gpu))
        checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar', map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint)
        net.eval()

        specs_dir = 'specs/' + str(args.expnum)
        if not os.path.exists(specs_dir):
            os.makedirs(specs_dir)

        with torch.no_grad():
            # tr
            count = 0
            eval_metric_total = 0
            #data_bar = tqdm(val_loader)
            #for input in data_bar:
            for _, input in enumerate(tqdm(train_loader)):
                loss, eval_metric = forward_common(istft, input, net, Loss, 'tr', args.loss_type, stride_product_time, expnum=args.expnum, mode='generate', Eval=Eval, fix_len_by_cl=args.fix_len_by_cl, count=count)
                reverb_paths = []
                for n in range(input[0].size(0)):
                    reverb_paths.append(input[6][n])
                count = count + 1
                sio.savemat('specs/' + str(args.expnum) + '/SDR_tr_' + str(count) + '.mat', {'SDR':eval_metric.data.cpu().numpy(), 'reverb_path': np.asarray(tuple(reverb_paths))})
                eval_metric_total += eval_metric.mean().item()
                if(count == args.nGenerate):
                    break
            eval_metric_total = eval_metric_total/count
            print('tr SDR = ' + str(eval_metric_total))

            # dt
            if(len(args.val_manifest) > 0):
                count = 0
                eval_metric_total = 0
                # data_bar = tqdm(val_loader)
                # for input in data_bar:
                for _, input in enumerate(tqdm(val_loader)):
                    loss, eval_metric = forward_common(istft, input, net, Loss, 'dt', args.loss_type, stride_product_time, expnum=args.expnum, mode='generate', Eval=Eval, fix_len_by_cl=args.fix_len_by_cl, count=count)
                    count = count + 1
                    reverb_paths = []
                    for n in range(input[0].size(0)):
                        reverb_paths.append(input[6][n])
                    sio.savemat('specs/' + str(args.expnum) + '/SDR_dt_' + str(count) + '.mat', {'SDR':eval_metric.data.cpu().numpy(), 'reverb_path': np.asarray(tuple(reverb_paths))})
                    eval_metric_total += eval_metric.mean().item()
                    if(count == args.nGenerate):
                        break
                eval_metric_total = eval_metric_total/count
                print('dt SDR = ' + str(eval_metric_total))

#def evaluate(istft, loader, net, Loss, data_type, loss_type, stride_product, logger, epoch, log_iter, expnum, Eval, fix_len_by_cl):
def evaluate(istft, loader, net, Loss, data_type, loss_type, stride_product, logger, epoch, Eval, fix_len_by_cl, w_var= 0, f_start_ratio=0, f_end_ratio=1):
    count = 0
    loss_total = 0
    loss_w_var_total = 0
    eval_metric_total = 0
    # data_bar = tqdm(loader)
    # for input in data_bar:
    for _, input in enumerate(tqdm(loader)):
        count += 1
        loss, eval_metric = forward_common(istft, input, net, Loss, data_type, loss_type, stride_product, mode='train', save_wav=False, Eval=Eval, fix_len_by_cl=fix_len_by_cl, w_var=w_var, f_start_ratio=f_start_ratio, f_end_ratio=f_end_ratio)
        if (w_var > 0):
            loss_w_var = loss[1]
            loss = loss[0]
            loss_w_var_total += loss_w_var.item()
        loss = torch.mean(loss)
        loss_total += loss.item()
        eval_metric = torch.mean(eval_metric)
        eval_metric_total += eval_metric.item()

    loss_total = loss_total / len(loader)
    eval_metric_total = eval_metric_total / len(loader)
    loss_w_var_total = loss_w_var_total / len(loader)
    print(data_type + ', epoch: ' + str(epoch) + ', loss: ' + str(loss_total))
    print(data_type + ', epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_total))
    print(data_type + ', epoch: ' + str(epoch) + ', w_var: ' + str(loss_w_var_total))
    logger.write(data_type + ', epoch: ' + str(epoch) + ', loss: ' + str(loss_total) + '\n')
    logger.write(data_type + ', epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_total) + '\n')
    logger.write(data_type + ', epoch: ' + str(epoch) + ', w_var: ' + str(loss_w_var_total) + '\n')

    logger.flush()

def evaluate_segment(interval_list, loader, net, Eval, data_type, stride_product, logger, epoch, fix_len_by_cl):
    nInterval = len(interval_list)-1
    # data_bar = tqdm(loader)
    # for input in data_bar:
    for _, input in enumerate(tqdm(loader)):
        eval_metric_interval = eval_segment(interval_list, input, net, Eval, stride_product, fix_len_by_cl= fix_len_by_cl)
        for n in range(nInterval):
            print(data_type + ', epoch: ' + str(epoch) + ', eval_metric (segment ' + str(n) + '): ' + str(eval_metric_interval[n]))
            logger.write(data_type + ', epoch: ' + str(epoch) + ', eval_metric (segment ' + str(n) + '): ' + str(eval_metric_interval[n]) + '\n')

if __name__ == '__main__':
    config, unparsed = get_config()

    main(config)
