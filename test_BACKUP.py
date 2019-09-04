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

from essential import forward_common
from config import get_config

import utils
from utils import get_stride_product_time
from models.unet import Unet
from models.layers.istft import ISTFT
from models.loss import wSDRLoss, cossim_time, cossim_spec, cossim_mag, sInvSDR_time, sInvSDR_from_loss
from se_dataset import AudioDataset
from torch.utils.data import DataLoader

import pdb

# TODO - loader clean speech tempo perturbed as input
# TODO - loader clean speech volume pertubed as input
# TODO - option for (tempo/volume/tempo+volume)
# TODO - loader noise sound as second input
# TODO - loader reverb effect as second input
# TODO - option for (noise/reverb/noise+reverb)

def main(args):
    assert (args.expnum >= 1)
    assert (args.gpu >= 0)

    if (args.cut_dtet):
        args.val_cut_manifest = 'data_sorted/reverb_dt_simu_8ch_cut_paired.csv'
        args.te_smallset_cut_manifest = 'data_sorted/reverb_et_hard6cases_simu_8ch_cut_paired.csv'

    torch.cuda.set_device(args.gpu)

    # n_fft, hop_length = 400, 160
    # n_fft, hop_length = 512, 256

    n_fft = args.n_fft
    hop_length = int(n_fft/2)
    window = torch.hann_window(n_fft).cuda()
    stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
    istft = ISTFT(n_fft, hop_length, window='hanning').cuda()

    json_path = os.path.join(args.model_json)
    params = utils.Params(json_path)

    net = Unet(params.model, loss_type=args.loss_type, nMic = args.nMic, reverb_frame=args.reverb_frame, use_bn=args.use_BN, input_type=args.input_type).cuda()
    checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar')
    net.load_state_dict(checkpoint)

    stride_product_time = get_stride_product_time(params.model['encoders'])

    if(args.eval_train):
        train_dataset = AudioDataset(manifest_path=args.tr_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_path=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate, shuffle=True, num_workers=0)

    if(args.eval_val):
        val_dataset = AudioDataset(manifest_path=args.val_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_path=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate, shuffle=False, num_workers=0)

    if(args.eval_test):
        test_dataset = AudioDataset(manifest_path=args.te_smallset_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_path=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate, shuffle=False, num_workers=0)

    if(args.cut_dtet):
        val_cut_dataset = AudioDataset(manifest_path=args.val_cut_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2, return_path=True)
        test_cut_dataset = AudioDataset(manifest_path=args.te_smallset_cut_manifest, nMic=args.nMic, sampling_method=args.mic_sampling, subset1=args.subset1, subset2=args.subset2)

        val_cut_loader = DataLoader(dataset=val_cut_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate, shuffle=False, num_workers=0, return_path=True)
        test_cut_loader = DataLoader(dataset=test_cut_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate, shuffle=False, num_workers=0)


    logger = open('logs/testlog_' + str(args.expnum) + '_' + args.eval_type + '.txt', 'w')
    if not os.path.exists('wavs/' + str(args.expnum) + '_eval'):
        os.makedirs('wavs/' + str(args.expnum) + '_eval')

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

    # Evaluation metric
    if(args.eval_type == 'time'):
        Eval = sInvSDR_time
    elif(args.eval_type == 'from_loss'):
        Eval = sInvSDR_from_loss
    else:
        Eval = None

    with torch.no_grad():
        net.eval()

        # Validation
        if(args.eval_val):
            evaluate(stft, istft, val_loader, net, Loss, 'val', args.loss_type, stride_product_time, logger, expnum = args.expnum, Eval=Eval)

        # Test
        if (args.eval_test):
            evaluate(stft, istft, test_loader, net, Loss, 'test', args.loss_type, stride_product_time, logger, expnum = args.expnum, Eval=Eval)

        # Train
        if (args.eval_train):
            evaluate(stft, istft, train_loader, net, Loss, 'train', args.loss_type, stride_product_time, logger, expnum = args.expnum, Eval=Eval)

        # Validation (cut) & Test (cut)
        if(args.cut_dtet):
            evaluate(stft, istft, val_cut_loader, net, Loss, 'val_cut', args.loss_type, stride_product_time, logger, expnum = args.expnum, Eval=Eval)
            evaluate(stft, istft, test_cut_loader, net, Loss, 'test_cut', args.loss_type, stride_product_time, logger, expnum = args.expnum, Eval=Eval)

    logger.close()

def evaluate(istft, loader, net, Loss, data_type, loss_type, stride_product, logger, expnum=-1, Eval=None):
    bar = tqdm(loader)
    count = 0
    loss_total = 0
    eval_metric_total = 0

    #eval_metric_multitime_max_total = 0
    #
    for input in bar:
        count += 1
        outputs = \
            forward_common(istft, input, net, Loss, data_type, loss_type, stride_product, mode='test', expnum=expnum, Eval=Eval)
        #loss, eval_metric, eval_metric_multitime_max, eval_metric_multitime_min = outputs[0], outputs[1], outputs[2], outputs[3]
        loss, eval_metric = outputs[0], outputs[1]

        loss = torch.mean(loss)
        loss_total += loss.item()

        if(Eval is not None):
            eval_metric = torch.mean(eval_metric)
            #eval_metric_multitime_max = torch.mean(eval_metric_multitime_max)
            #eval_metric_multitime_min = torch.mean(eval_metric_multitime_min)
            eval_metric_total += eval_metric.item()
            #eval_metric_multitime_min_total += eval_metric_multitime_min.item()
            #eval_metric_multitime_max_total += eval_metric_multitime_max.item()

    loss_total = loss_total / len(loader)
    eval_metric_total = eval_metric_total / len(loader)
    #eval_metric_multitime_min_total = eval_metric_multitime_min_total / len(loader)
    #eval_metric_multitime_max_total = eval_metric_multitime_max_total / len(loader)

    #print(data_type + ', loss: ' + str(loss_total) + ', eval_metric = ' + str(eval_metric_total) + ', eval_metric (shift) = ' + str(eval_metric_multitime_min_total) + ' ~ ' + str(eval_metric_multitime_max_total))
    #logger.write(data_type + ', loss: ' + str(loss_total) + ', eval_metric = ' + str(eval_metric_total) + ', eval_metric (shift) = ' + str(eval_metric_multitime_min_total) + ' ~ ' + str(eval_metric_multitime_max_total) + '\n')

    print(data_type + ', loss: ' + str(loss_total) + ', eval_metric = ' + str(eval_metric_total))
    logger.write(data_type + ', loss: ' + str(loss_total) + ', eval_metric = ' + str(eval_metric_total) + '\n')

    logger.flush()


if __name__ == '__main__':
    config, unparsed = get_config()

    main(config)

