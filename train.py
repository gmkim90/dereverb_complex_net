import os
import sys
import torch
import numpy as np

import torch.optim as optim

from tqdm import tqdm
import gc

import utils
from utils import get_stride_product_time, count_parameters, save_input_mat_for_debug
from shutil import copyfile
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
    assert (args.expnum >= 1)

    # Get free gpu
    gpuIdx = utils.get_freer_gpu()
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpuIdx)
    print('gpuIdx = ' + str(gpuIdx) + ' is selected')

    # define FFT (we don't need window here)
    n_fft = args.nFFT

    if(args.mode == 'generate'):
        args.return_path = True
        shuffle_train_loader = False
    else:
        shuffle_train_loader = True

    src_range_list = args.src_range.replace("'", "").split(',')
    src_range_list = [float(p) for p in src_range_list] # convert str to float
    assert (len(src_range_list) == 6)

    if (len(args.tr_manifest) > 0):
        if(args.tr_manifest.find('data_sorted') == -1):
            args.tr_manifest = 'data_sorted/' + args.tr_manifest
        train_dataset = SpecDataset(manifest_path=args.tr_manifest, nMic=args.nMic, return_path=args.return_path,
                                    src_range=src_range_list, start_ratio=args.start_ratio, end_ratio=args.end_ratio,
                                    interval_cm=args.interval_cm_tr, use_ref_IR=args.use_ref_IR)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate, shuffle=shuffle_train_loader, num_workers=0)

    if (len(args.trsub_manifest) > 0):
        if (args.trsub_manifest.find('data_sorted') == -1):
            args.trsub_manifest = 'data_sorted/' + args.trsub_manifest
        trsub_dataset = SpecDataset(manifest_path=args.trsub_manifest, nMic=args.nMic, return_path=args.return_path,
                                    src_range=src_range_list, start_ratio=args.start_ratio, end_ratio=args.end_ratio,
                                    interval_cm=args.interval_cm_tr, use_ref_IR=args.use_ref_IR_te)
        trsub_loader = DataLoader(dataset=trsub_dataset, batch_size=args.batch_size, collate_fn=trsub_dataset.collate, shuffle=False, num_workers=0)

    if (len(args.val_manifest) > 0):
        if (args.val_manifest.find('data_sorted') == -1):
            args.val_manifest = 'data_sorted/' + args.val_manifest
        val_dataset = SpecDataset(manifest_path=args.val_manifest, nMic=args.nMic, return_path=args.return_path,
                                  src_range='all', interval_cm=args.interval_cm_te, use_ref_IR=args.use_ref_IR_te)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate, shuffle=False, num_workers=0)

    if(len(args.te1_manifest) > 0):
        if (args.te1_manifest.find('data_sorted') == -1):
            args.te1_manifest = 'data_sorted/' + args.te1_manifest
        test1_dataset = SpecDataset(manifest_path=args.te1_manifest, nMic=args.nMic,
                                    src_range='all', interval_cm=args.interval_cm_te, use_ref_IR=args.use_ref_IR_te)
        test1_loader = DataLoader(dataset=test1_dataset, batch_size=args.batch_size, collate_fn=test1_dataset.collate, shuffle=False, num_workers=0)

    if(len(args.te2_manifest) > 0):
        if (args.te2_manifest.find('data_sorted') == -1):
            args.te2_manifest = 'data_sorted/' + args.te2_manifest
        test2_dataset = SpecDataset(manifest_path=args.te2_manifest, nMic=args.nMic,
                                    src_range='all', interval_cm=args.interval_cm_te, use_ref_IR=args.use_ref_IR_te)
        test2_loader = DataLoader(dataset=test2_dataset, batch_size=args.batch_size, collate_fn=test2_dataset.collate, shuffle=False, num_workers=0)

    torch.set_printoptions(precision=10, profile="full")

    # Set loss type
    if(len(args.loss_type) > 0):
        if(args.loss_type.find('positive') == -1):
            Loss = getattr(losses, args.loss_type)
        else:
            Loss = getattr(losses, args.loss_type.replace('_positive', ''))
    else:
        Loss = None

    if(len(args.loss2_type) > 0):
        if(args.loss2_type.find('positive') == -1):
            Loss2 = getattr(losses, args.loss2_type)
        else:
            Loss2 = getattr(losses, args.loss2_type.replace('_positive', ''))
    else:
        Loss2 = None

    if(len(args.eval_type) > 0):
        if(args.eval_type.find('positive') == -1):
            Eval = getattr(losses, args.eval_type)
        else:
            Eval = getattr(losses, args.eval_type.replace('_positive', ''))
    else:
        Eval = None

    if (len(args.eval2_type) > 0):
        if(args.eval2_type.find('positive') == -1):
            Eval2 = getattr(losses, args.eval2_type)
        else:
            Eval2 = getattr(losses, args.eval2_type.replace('_positive', ''))
    else:
        Eval2 = None

    # Network
    if(args.model_type == 'unet'):
        from models.unet import Unet
        if (args.model_json.find('models') == -1):
            args.model_json = 'models/' + args.model_json
        json_path = os.path.join(args.model_json)
        params = utils.Params(json_path)
        net = Unet(params.model, nMic = args.nMic,
                    input_type=args.input_type, ds_rate = args.ds_rate, w_init_std=args.w_init_std)
    elif(args.model_type == 'cMLP'):
        from models.cMLP import cMLP
        net = cMLP(nLayer = args.nLayer, nHidden = args.nHidden, nFreq = args.nFreq, nMic = args.nMic, ds_rate = args.ds_rate)


    if(args.mode == 'train'):
        loss_valmin = 1000000000
        maxiter = (args.end_epoch - args.start_epoch)*len(train_loader)
        if (args.eval_iter == 0 or args.eval_iter > maxiter):
            args.eval_iter = maxiter

        if (args.log_iter == 0 or args.log_iter > maxiter):
            args.log_iter = maxiter


        if(args.start_epoch > 0):
            print('training starts from epoch '+ str(args.start_epoch))

            checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model.pth.tar',
                                    map_location=lambda storage, loc: storage)

            if('loss_valmin' in checkpoint.keys()):
                loss_valmin = checkpoint['loss_valmin']

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

        # count parameter size
        nParam = count_parameters(net)
        print('#param = ' + str(nParam))

        # Learning rate scheduler
        #scheduler = ExponentialLR(optimizer, 0.95) # do not use scheduler anymore

        count = 0
        count_mb = 0
        count_eval = 0

        for epoch in range(args.start_epoch, args.end_epoch):
            # train
            loss_mb = 0
            loss2_mb = 0
            eval_metric_mb = 0
            eval2_metric_mb = 0

            for _, input in enumerate(tqdm(train_loader)):
                count += 1
                count_mb += 1

                if(not count % args.log_iter == 0):
                    loss, loss2, eval_metric, eval2_metric = \
                        forward_common(input, net, Loss,
                                       Eval=Eval, Eval2=Eval2, Loss2 = Loss2,
                                       loss_type = args.loss_type, loss2_type=args.loss2_type, eval_type=args.eval_type, eval2_type=args.eval2_type,
                                         use_ref_IR=args.use_ref_IR)
                    loss_mean = torch.mean(loss)
                    if(torch.isnan(loss_mean).item()):
                        print('NaN is detected on loss, terminate program')
                        logger.write('NaN is detected on loss, terminate program' + '\n')
                        sys.exit()
                    loss_mb += loss_mean.item()
                    if(loss2 is not None):
                        loss2_mean = torch.mean(loss2)
                        loss2_mb += float(loss2_mean.item())
                    if(eval_metric is not None):
                        eval_metric_mean = torch.mean(eval_metric).item()
                        eval_metric_mb += float(eval_metric_mean)
                    if(eval2_metric is not None):
                        eval2_metric_mean = torch.mean(eval2_metric).item()
                        eval2_metric_mb += float(eval2_metric_mean)
                else:
                    if(args.save_input_mat_for_debug):
                        save_input_mat_for_debug(input, count)
                    loss, loss2, eval_metric, eval2_metric = \
                        forward_common(input, net, Loss,
                                       Loss2=Loss2, Eval=Eval, Eval2=Eval2,
                                       loss_type = args.loss_type, loss2_type=args.loss2_type, eval_type=args.eval_type,eval2_type=args.eval2_type,
                                       use_ref_IR = args.use_ref_IR)
                    loss_mean = torch.mean(loss)
                    if(torch.isnan(loss_mean).item()):
                        print('NaN is detected on loss, terminate program')
                        logger.write('NaN is detected on loss, terminate program' + '\n')
                        sys.exit()
                    loss_mb += loss_mean.item()
                    print('train, epoch: ' + str(epoch) + ', loss: ' + str(loss_mean.item()))
                    logger.write('train, epoch: ' + str(epoch) + ', loss: ' + str(loss_mean.item()) + '\n')

                    if(loss2 is not None):
                        loss2_mean = torch.mean(loss2)
                        loss2_mb += float(loss2_mean.item())
                        print('train, epoch: ' + str(epoch) + ', loss2: ' + str(loss2_mean.item()))
                        logger.write('train, epoch: ' + str(epoch) + ', loss2: ' + str(loss2_mean.item()) + '\n')
                    if(eval_metric is not None):
                        eval_metric_mean = torch.mean(eval_metric).item()
                        eval_metric_mb += float(eval_metric_mean)
                        print('train, epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_mean))
                        logger.write('train, epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_mean) + '\n')
                    if(eval2_metric is not None):
                        eval2_metric_mean = torch.mean(eval2_metric).item()
                        eval2_metric_mb += float(eval2_metric_mean)
                        print('train, epoch: ' + str(epoch) + ', eval2_metric: ' + str(eval2_metric_mean))
                        logger.write('train, epoch: ' + str(epoch) + ', eval2_metric: ' + str(eval2_metric_mean) + '\n')

                    loss_mb = loss_mb/count_mb
                    loss2_mb = loss2_mb / count_mb
                    eval_metric_mb = eval_metric_mb/count_mb
                    eval2_metric_mb = eval2_metric_mb/count_mb

                    print('train, epoch: ' + str(epoch) + ', loss (minibatch): ' + str(loss_mb))
                    print('train, epoch: ' + str(epoch) + ', loss2 (minibatch): ' + str(loss2_mb))
                    print('train, epoch: ' + str(epoch) + ', eval_metric (minibatch): ' + str(eval_metric_mb))
                    print('train, epoch: ' + str(epoch) + ', eval2_metric (minibatch): ' + str(eval2_metric_mb))

                    logger.write('train, epoch: ' + str(epoch) + ', loss: (minibatch): ' + str(loss_mb) + '\n')
                    logger.write('train, epoch: ' + str(epoch) + ', loss2: (minibatch): ' + str(loss2_mb) + '\n')
                    logger.write('train, epoch: ' + str(epoch) + ', eval_metric: (minibatch): ' + str(eval_metric_mb) + '\n')
                    logger.write('train, epoch: ' + str(epoch) + ', eval2_metric: (minibatch): ' + str(eval2_metric_mb) + '\n')
                    logger.flush()

                    loss_mb = 0
                    loss2_mb = 0
                    eval_metric_mb = 0
                    eval2_metric_mb = 0
                    count_mb = 0


                #utils.CPUmemDebug('before backward & step', mem_debug_file)

                optimizer.zero_grad()
                if(loss2 is not None and args.w_loss2 > 0):
                    loss_mean += loss2_mean*args.w_loss2
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
                            evaluate(trsub_loader, net, Loss, 'trsub',
                                     logger, epoch, Eval, Eval2, Loss2,
                                     loss_type = args.loss_type, eval_type = args.eval_type
                                     ,eval2_type=args.eval2_type, loss2_type=args.loss2_type, use_ref_IR=args.use_ref_IR_te)  # do not use Loss2 for evaluate

                        # Validaion
                        if (len(args.val_manifest) > 0):
                            loss_val_total = evaluate(val_loader, net, Loss, 'val',
                                     logger, epoch, Eval, Eval2, Loss2,
                                     loss_type=args.loss_type, eval_type = args.eval_type
                                     ,eval2_type=args.eval2_type, loss2_type=args.loss2_type, use_ref_IR=args.use_ref_IR_te)  # do not use Loss2 for evaluate

                        # Test
                        if (len(args.te1_manifest) > 0):
                            evaluate(test1_loader, net, Loss, 'te1',
                                     logger, epoch, Eval, Eval2, Loss2,
                                     loss_type=args.loss_type, eval_type = args.eval_type ,eval2_type=args.eval2_type, loss2_type=args.loss2_type, use_ref_IR=args.use_ref_IR_te)  # do not use Loss2 for evaluate


                        # Test2
                        if (len(args.te2_manifest) > 0):
                            evaluate(test2_loader, net, Loss, 'te2',
                                     logger, epoch, Eval, Eval2, Loss2,
                                     loss_type=args.loss_type, eval_type = args.eval_type ,eval2_type=args.eval2_type, loss2_type=args.loss2_type, use_ref_IR=args.use_ref_IR_te)  # do not use Loss2 for evaluate

                        net.train()
                        gc.collect()
                        utils.CPUmemDebug('memory after gc.collect()', logger)

                    torch.save({'epoch': epoch+1, 'model':net.state_dict(), 'optimizer': optimizer.state_dict(),
                                'loss_val':loss_val_total, 'loss_valmin':loss_valmin},
                               'checkpoint/' + str(args.expnum) + '-model.pth.tar')

                    if(loss_valmin > loss_val_total):
                        loss_valmin = loss_val_total
                        #print('valid loss decrease compared to min, save valmin model')
                        #copyfile('checkpoint/' + str(args.expnum) + '-model.pth.tar', 'checkpoint/' + str(args.expnum) + '-model-valmin.pth.tar')
                        torch.save({'epoch': epoch + 1, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'loss_val': loss_val_total, 'loss_valmin': loss_valmin},
                                   'checkpoint/' + str(args.expnum) + '-model-valmin.pth.tar')

            #torch.save({'epoch': epoch + 1, 'model': net.state_dict(), 'optimizer': optimizer.state_dict()},
#                       'checkpoint/' + str(args.expnum) + '-model.pth.tar')
            torch.cuda.empty_cache()
        logger.close()

    elif(args.mode == 'generate'):
        print('load valmin model')

        checkpoint = torch.load('checkpoint/' + str(args.expnum) + '-model-valmin.pth.tar', map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['model'])
        net.cuda()

        del checkpoint
        torch.cuda.empty_cache()

        if not os.path.exists('specs/' + str(args.expnum)):
            os.makedirs('specs/' + str(args.expnum))


        with torch.no_grad():
            if (len(args.tr_manifest) > 0): # tr
                count = 0
                loss_total = 0
                loss2_total = 0
                eval_metric_total = 0
                eval2_metric_total = 0
                for _, input in enumerate(tqdm(train_loader)):
                    count = count + 1

                    savename = 'specs/' + str(args.expnum) + '/tr_' + str(count) + '.mat'

                    loss, loss2, eval_metric, eval2_metric = \
                        forward_common(input, net, Loss, Eval=Eval, Eval2=Eval2, Loss2 = Loss2,
                                       loss_type = args.loss_type, loss2_type=args.loss2_type, eval_type=args.eval_type, eval2_type=args.eval2_type,
                                         use_ref_IR=args.use_ref_IR, save_activation=args.save_activation, savename=savename)

                    metrics_save = {}
                    if(loss is not None):
                        loss_total += loss.mean().item()
                        metrics_save['loss'] = loss.data.cpu().numpy()
                    if(loss2 is not None):
                        loss2_total += loss2.mean().item()
                        metrics_save['loss2'] = loss2.data.cpu().numpy()
                    if(eval_metric is not None):
                        eval_metric_total += eval_metric.mean().item()
                        metrics_save['eval_metric'] = eval_metric.data.cpu().numpy()
                    if(eval2_metric is not None):
                        eval2_metric_total += eval2_metric.mean().item()
                        metrics_save['eval2_metric'] = eval2_metric.data.cpu().numpy()

                    reverb_paths = []
                    for n in range(input[0].size(0)):
                        reverb_paths.append(input[2][n])
                    metrics_save['reverb_paths'] = np.asarray(tuple(reverb_paths))

                    specs_path = 'specs/' + str(args.expnum) + '/metric_tr_' + str(count) + '.mat'

                    sio.savemat(specs_path, metrics_save)

                print('tr loss = ' + str(loss_total))
                print('tr loss2 = ' + str(loss2_total))
                print('tr eval = ' + str(eval_metric_total))
                print('tr eval2 = ' + str(eval2_metric_total))
            else:
                print("NO TRAINING MANIFEST")

            if (len(args.val_manifest) > 0): # tr
                count = 0
                loss_total = 0
                loss2_total = 0
                eval_metric_total = 0
                eval2_metric_total = 0
                for _, input in enumerate(tqdm(val_loader)):
                    count = count + 1

                    savename = 'specs/' + str(args.expnum) + '/val_' + str(count) + '.mat'

                    loss, loss2, eval_metric, eval2_metric = \
                        forward_common(input, net, Loss, Eval=Eval, Eval2=Eval2, Loss2 = Loss2,
                                       loss_type = args.loss_type, loss2_type=args.loss2_type, eval_type=args.eval_type, eval2_type=args.eval2_type,
                                         use_ref_IR=args.use_ref_IR_te, save_activation=args.save_activation, savename=savename)

                    metrics_save = {}
                    if(loss is not None):
                        loss_total += loss.mean().item()
                        metrics_save['loss'] = loss.data.cpu().numpy()
                    if(loss2 is not None):
                        loss2_total += loss2.mean().item()
                        metrics_save['loss2'] = loss2.data.cpu().numpy()
                    if(eval_metric is not None):
                        eval_metric_total += eval_metric.mean().item()
                        metrics_save['eval_metric'] = eval_metric.data.cpu().numpy()
                    if(eval2_metric is not None):
                        eval2_metric_total += eval2_metric.mean().item()
                        metrics_save['eval2_metric'] = eval2_metric.data.cpu().numpy()

                    reverb_paths = []
                    for n in range(input[0].size(0)):
                        reverb_paths.append(input[2][n])
                    metrics_save['reverb_paths'] = np.asarray(tuple(reverb_paths))

                    specs_path = 'specs/' + str(args.expnum) + '/metric_val_' + str(count) + '.mat'

                    sio.savemat(specs_path, metrics_save)

                print('val loss = ' + str(loss_total))
                print('val loss2 = ' + str(loss2_total))
                print('val eval = ' + str(eval_metric_total))
                print('val eval2 = ' + str(eval2_metric_total))
            else:
                print("NO VALID MANIFEST")

def evaluate(loader, net, Loss, data_type,
             logger, epoch,  Eval, Eval2, Loss2,
             eval_type = '', eval2_type='', loss_type = '', loss2_type ='',
             use_ref_IR=False):
    count = 0
    loss_total = 0
    loss2_total = 0
    eval_metric_total = 0
    eval2_metric_total = 0

    with torch.no_grad():
        for _, input in enumerate(tqdm(loader)):
            count += 1
            loss, loss2, eval_metric, eval2_metric = forward_common(input, net, Loss, Eval=Eval, Eval2=Eval2, Loss2=Loss2,
                                                                    loss_type = loss_type, eval_type=eval_type, eval2_type=eval2_type, loss2_type=loss2_type,
                                                                    use_ref_IR=use_ref_IR) # do not use Loss2 & ref_IR for eval

            if(loss is not None):
                loss_mean = torch.mean(loss)
                loss_total += loss_mean.item()
            if(loss2 is not None):
                loss2_mean = torch.mean(loss2)
                loss2_total += loss2_mean.item()
            if(eval_metric is not None):
                eval_metric_mean = torch.mean(eval_metric)
                eval_metric_total += eval_metric_mean.item()
            if(eval2_metric is not None):
                eval2_metric_mean = torch.mean(eval2_metric)
                eval2_metric_total += eval2_metric_mean.item()

    loss_total = loss_total / len(loader)
    loss2_total = loss2_total / len(loader)
    eval_metric_total = eval_metric_total / len(loader)
    eval2_metric_total = eval2_metric_total / len(loader)

    print(data_type + ', epoch: ' + str(epoch) + ', loss: ' + str(loss_total))
    print(data_type + ', epoch: ' + str(epoch) + ', loss2: ' + str(loss2_total))
    print(data_type + ', epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_total))
    print(data_type + ', epoch: ' + str(epoch) + ', eval2_metric: ' + str(eval2_metric_total))

    if(logger is not None):
        logger.write(data_type + ', epoch: ' + str(epoch) + ', loss: ' + str(loss_total) + '\n')
        logger.write(data_type + ', epoch: ' + str(epoch) + ', loss2: ' + str(loss2_total) + '\n')
        logger.write(data_type + ', epoch: ' + str(epoch) + ', eval_metric: ' + str(eval_metric_total) + '\n')
        logger.write(data_type + ', epoch: ' + str(epoch) + ', eval2_metric: ' + str(eval2_metric_total) + '\n')

        logger.flush()

    return loss_total

if __name__ == '__main__':
    config, unparsed = get_config()

    main(config)
