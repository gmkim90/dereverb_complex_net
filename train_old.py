import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from scipy.io import wavfile
import librosa
from tqdm import tqdm
import math

import utils
from utils import get_stride_product_time
from models.unet import Unet
from models.layers.istft import ISTFT
from models.loss import cossim_time, cossim_spec, cossim_mag, sInvSDR_time, sInvSDR_spec, negative_MSE, sInvSDR_mag
from se_dataset import SpecDataset
from torch.utils.data import DataLoader

from essential import forward_common
from config import get_config

import pdb

# TODO - loader clean speech tempo perturbed as input
# TODO - loader clean speech volume pertubed as input
# TODO - option for (tempo/volume/tempo+volume)
# TODO - loader noise sound as second input
# TODO - loader reverb effect as second input
# TODO - option for (noise/reverb/noise+reverb)

# n_fft, hop_length = 400, 160
#n_fft, hop_length = 512, 256

def main(args):
    assert (args.expnum >= 1)
    assert (args.gpu >= 0)
    if (args.nMic == 8):
        assert (args.mic_sampling == 'no')
    if (args.mic_sampling == 'no'):
        assert (args.nMic == 8)
    if (args.mic_sampling == 'ref_manual'):
        args.subset1 = args.subset1.split(',')  # string to list
        args.subset2 = args.subset2.split(',')  # string to list

    prefix = 'data_sorted/'
    if(args.td_type == 'gt'):
        args.tr_manifest = prefix + 'reverb_tr_simu_8ch_td_gt.csv'
        args.val_manifest = prefix +  'reverb_dt_simu_8ch_td_gt.csv'
        args.te_smallset_manifest = prefix + 'reverb_et_hard6cases_simu_8ch_td_gt.csv'
        args.val_trainIR_manifest = prefix + 'reverb_dt_simu_8ch_trainIR_td_gt.csv'
        args.te_smallset_trainIR_manifest = prefix + 'reverb_et_hard6cases_simu_8ch_trainIR_td_gt.csv'
        return_time_delay=True
    elif(args.td_type == 'est'):
        args.tr_manifest = prefix + 'reverb_tr_simu_8ch_td_est.csv'
        args.val_manifest = prefix +  'reverb_dt_simu_8ch_td_est.csv'
        args.te_smallset_manifest = prefix + 'reverb_et_hard6cases_simu_8ch_td_est.csv'
        args.val_trainIR_manifest = prefix + 'reverb_dt_simu_8ch_trainIR_td_est.csv'
        args.te_smallset_trainIR_manifest = prefix + 'reverb_et_hard6cases_simu_8ch_trainIR_td_est.csv'
        return_time_delay = True
    else:
        if(args.apply_vad_preemp):
            args.tr_manifest = prefix + 'reverb_tr_simu_vad_preemp.csv'
            args.val_manifest = prefix + 'reverb_dt_simu_vad_preemp.csv'
            args.te_smallset_manifest = prefix + 'reverb_et_hard6cases_simu_8ch_td_est.csv'
            #args.val_trainIR_manifest = prefix + 'reverb_dt_simu_8ch_trainIR_td_est.csv'
            #args.te_smallset_trainIR_manifest = prefix + 'reverb_et_hard6cases_simu_8ch_trainIR_td_est.csv'
            args.val_trainIR_manifest = '' # empty
            args.te_smallset_trainIR_manifest = '' # empty

        return_time_delay = False

    if(return_time_delay):
        assert(not args.mic_sampling == 'random'), 'random mic sampling cannot use time delay for now'

    #if (args.cut_dtet):
#        args.val_cut_manifest = 'data_sorted/reverb_dt_simu_8ch_cut_paired.csv'
#        args.te_smallset_cut_manifest = 'data_sorted/reverb_et_hard6cases_simu_8ch_cut_paired.csv'

    torch.cuda.set_device(args.gpu)

    #print('1')
    n_fft = args.nFFT
    win_size = args.nWin
    hop_length = int(win_size/2)
    window = torch.hann_window(win_size).cuda()
    stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
    istft = ISTFT(n_fft, hop_length, window='hanning').cuda()

    #print('2')

    # TODO - check exists
    #checkpoint = torch.load('./final.pth.tar')
    #net.load_state_dict(checkpoint)

    #train_dataset = AudioDataset(manifest_path=args.tr_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2)
    #val_dataset = AudioDataset(manifest_path=args.val_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2)
    #test_dataset = AudioDataset(manifest_path=args.te_smallset_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2)

    train_dataset = SpecDataset(manifest_path=args.tr_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_timedelay=return_time_delay)
    val_dataset = SpecDataset(manifest_path=args.val_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_timedelay=return_time_delay)
    test_dataset = SpecDataset(manifest_path=args.te_smallset_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_timedelay=return_time_delay)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate, shuffle=False, num_workers=0)

    #print('3')
    if(args.cut_dtet):
        #val_cut_dataset = AudioDataset(manifest_path=args.val_cut_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2)
        #test_cut_dataset = AudioDataset(manifest_path=args.te_smallset_cut_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2)

        val_cut_dataset = SpecDataset(manifest_path=args.val_cut_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_timedelay=return_time_delay)
        test_cut_dataset = SpecDataset(manifest_path=args.te_smallset_cut_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_timedelay=return_time_delay)

        val_cut_loader = DataLoader(dataset=val_cut_dataset, batch_size=args.batch_size, collate_fn=val_cut_dataset.collate, shuffle=False, num_workers=0)
        test_cut_loader = DataLoader(dataset=test_cut_dataset, batch_size=args.batch_size, collate_fn=test_cut_dataset.collate, shuffle=False, num_workers=0)

    #val_trainIR_dataset = AudioDataset(manifest_path=args.val_trainIR_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2)
    #test_trainIR_dataset = AudioDataset(manifest_path=args.te_smallset_trainIR_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2)


    val_trainIR_dataset = SpecDataset(manifest_path=args.val_trainIR_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_timedelay=return_time_delay)
    test_trainIR_dataset = SpecDataset(manifest_path=args.te_smallset_trainIR_manifest, stft=stft, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_timedelay=return_time_delay)

    val_trainIR_loader = DataLoader(dataset=val_trainIR_dataset, batch_size=args.batch_size, collate_fn = val_trainIR_dataset.collate, shuffle=False, num_workers=0)
    test_trainIR_loader = DataLoader(dataset=test_trainIR_dataset, batch_size=args.batch_size, collate_fn = test_trainIR_dataset.collate, shuffle=False, num_workers=0)

    #print('4')
    logger = open('logs/log_' + str(args.expnum) + '.txt', 'w')
    if not os.path.exists('wavs/' + str(args.expnum)):
        os.makedirs('wavs/' + str(args.expnum))

    torch.set_printoptions(precision=10, profile="full")
    #print('4-2')
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

    #print('5')
    # measure reverberant speech loss
    if(args.measure_reverb_loss):
        print('@epoch0: measuring reverberant training loss')
        rev_tr = measure_reverb_loss(train_loader, Loss, args.loss_type)
        print('rev_tr = ' + str(rev_tr))
        logger.write('rev_tr = ' + str(rev_tr) + '\n')

        print('@epoch0: measuring reverberant dt loss')
        rev_dt = measure_reverb_loss(val_loader, Loss, args.loss_type)
        print('rev_dt = ' + str(rev_dt))
        logger.write('rev_dt = ' + str(rev_dt) + '\n')

        #print('@epoch0: measuring reverberant et loss')
        #rev_et = measure_reverb_loss(test_loader, Loss, args.loss_type)
        #print('rev_et = ' + str(rev_et))
        #logger.write('rev_et = ' + str(rev_et) + '\n')

    # Network
    json_path = os.path.join(args.model_json)
    params = utils.Params(json_path)
    net = Unet(params.model, loss_type=args.loss_type, nMic = args.nMic, reverb_frame=args.reverb_frame, use_bn=args.use_BN, input_type=args.input_type, ds_rate = args.ds_rate).cuda()
    stride_product_time = get_stride_product_time(params.model['encoders'])

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Learning rate scheduler
    scheduler = ExponentialLR(optimizer, 0.95)

    for epoch in range(args.num_epochs):
        # train
        train_bar = tqdm(train_loader)
        count = 0
        count_mb = 0
        loss_mb = 0
        eval_metric_mb = 0
        for input in train_bar:
            count += 1
            count_mb += 1
            if(not count % args.log_iter == 0):
                loss, eval_metric = forward_common(istft, input, net, Loss, 'train', args.loss_type, stride_product_time, mode='train', Eval=Eval)
                loss = torch.mean(loss)
                loss_mb += loss.item()
                eval_metric = torch.mean(eval_metric)
                eval_metric_mb += eval_metric.item()
            else:
                loss, eval_metric = forward_common(istft, input, net, Loss, 'train', args.loss_type, stride_product_time, mode='train', save_wav=True, expnum=args.expnum, Eval=Eval)
                loss = torch.mean(loss)
                eval_metric = torch.mean(eval_metric)
                print('train, epoch: ' + str(epoch) + ', loss: ' + str(loss.item()))
                logger.write('train, epoch: ' + str(epoch) + ', loss: ' + str(loss.item()) + '\n')
                print('train, epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric.item()))
                logger.write('train, epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric.item()) + '\n')

                loss_mb = loss_mb/count_mb
                eval_metric_mb = eval_metric_mb/count_mb
                print('train, epoch: ' + str(epoch) + ', loss (minibatch): ' + str(loss_mb))
                logger.write('train, epoch: ' + str(epoch) + ', loss: (minibatch): ' + str(loss_mb) + '\n')
                print('train, epoch: ' + str(epoch) + ', eval_metric (minibatch): ' + str(eval_metric_mb))
                logger.write('train, epoch: ' + str(epoch) + ', eval_metric: (minibatch): ' + str(eval_metric_mb) + '\n')

                loss_mb = 0
                eval_metric_mb = 0
                count_mb = 0
                logger.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if(args.do_eval):
                net.eval()

            # Validation
            evaluate(istft, val_loader, net, Loss, 'val', args.loss_type, stride_product_time, logger, epoch, args.log_iter, args.expnum, Eval)

            # Test
            evaluate(istft, test_loader, net, Loss, 'test', args.loss_type, stride_product_time, logger, epoch, args.log_iter, args.expnum, Eval)

            # Validation (cut) & Test (cut)
            if(args.cut_dtet):
                evaluate(istft, val_cut_loader, net, Loss, 'val_cut', args.loss_type, stride_product_time, logger, epoch, args.log_iter, args.expnum, Eval)
                evaluate(istft, test_cut_loader, net, Loss, 'test_cut', args.loss_type, stride_product_time, logger, epoch, args.log_iter, args.expnum, Eval)

            # Validation (trainIR) & Test (trainIR)
            evaluate(istft, val_trainIR_loader, net, Loss, 'val_trainIR', args.loss_type, stride_product_time, logger, epoch, args.log_iter, args.expnum, Eval)

            # Test
            evaluate(istft, test_trainIR_loader, net, Loss, 'test_trainIR', args.loss_type, stride_product_time, logger, epoch, args.log_iter, args.expnum, Eval)

            net.train()
        torch.save(net.state_dict(), 'checkpoint/' + str(args.expnum) + '-model.pth.tar')
    logger.close()

def evaluate(istft, loader, net, Loss, data_type, loss_type, stride_product, logger, epoch, log_iter, expnum, Eval):
    bar = tqdm(loader)
    count = 0
    loss_total = 0
    eval_metric_total = 0
    for input in bar:
        count += 1
        if (not count % log_iter == 0):
            loss, eval_metric = forward_common(istft, input, net, Loss, data_type, loss_type, stride_product, mode='train', save_wav=False, Eval=Eval)
        else:
            loss, eval_metric = forward_common(istft, input, net, Loss, data_type, loss_type, stride_product, mode='train', expnum=expnum, save_wav=True, Eval=Eval)
            print(data_type + ', epoch: ' + str(epoch) + ', loss (wav): ' + str(loss[0].item()))
            print(data_type + ', epoch: ' + str(epoch) + ', eval_metric (wav): ' + str(eval_metric[0].item()))
            logger.write(data_type + ', epoch: ' + str(epoch) + ', loss (wav): ' + str(loss[0].item()) + '\n')
            logger.write(data_type + ', epoch: ' + str(epoch) + ', eval_metric (wav): ' + str(eval_metric[0].item()) + '\n')

        loss = torch.mean(loss)
        loss_total += loss.item()
        eval_metric = torch.mean(eval_metric)
        eval_metric_total += eval_metric.item()

    loss_total = loss_total / len(loader)
    eval_metric_total = eval_metric_total / len(loader)
    print(data_type + ', epoch: ' + str(epoch) + ', loss: ' + str(loss_total))
    print(data_type + ', epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_total))
    logger.write(data_type + ', epoch: ' + str(epoch) + ', loss: ' + str(loss_total) + '\n')
    logger.write(data_type + ', epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_total) + '\n')
    logger.flush()

def measure_reverb_loss(loader, Loss, loss_type):
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

if __name__ == '__main__':
    config, unparsed = get_config()

    main(config)
