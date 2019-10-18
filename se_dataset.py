import numpy as np

import soundfile as sf # alternative to torchaudio
import torch
from torch.utils import data
import random
import math
import os

import time
import pdb

#def check_valid_position(idx_to_pos, pos_str_splited, pos_range):
def check_valid_position(pos_val, pos_range, interval_cm=1):
    # pos_range : 3x2 (3 = x,y,z, 2 = min/max)
    pos_x = pos_val[0]
    pos_y = pos_val[1]
    pos_z = pos_val[2]

    if(pos_x >= pos_range[0] and pos_x <= pos_range[1] and pos_y >= pos_range[2] and pos_y <= pos_range[3]
    and pos_z >= pos_range[4] and pos_z <= pos_range[5]): # pos_range : 1D
        if(interval_cm > 1):
            #pdb.set_trace()
            if(round(pos_x * 100) % interval_cm == 0 and
                round(pos_y * 100) % interval_cm == 0 and
                round(pos_y * 100) % interval_cm == 0):
                return True, [pos_x, pos_y, pos_z]
            else:
                return False, None
        else:
            return True, [pos_x, pos_y, pos_z]

    return False, None

def load_data_list(manifest_path='', src_range = None, use_ref_IR=False, interval_cm=1, start_ratio = 0, end_ratio = 1):
    # ref IR 안 쓰는 실험도 할 수 있도록 해야지

    dataset = {}
    manifest = open(manifest_path, 'r')
    lines = manifest.readlines()
    valid_input = True

    if (src_range == 'all'):  # special case : all position within room is allowed
        src_pos_range = [0, 10000, 0, 10000, 0, 10000] # large enough room
    else:
        src_pos_range = src_range


    print("Loading files from manifst " + manifest_path)
    dataset['tarIR'] = []
    if(use_ref_IR):
        dataset['refIR'] = []

    nLine = len(lines)
    startIdx = math.floor(start_ratio*nLine)
    endIdx = math.floor(end_ratio*nLine)

    #for line in lines:
    for i in range(startIdx, endIdx):
        line = lines[i].strip()
        line_splited = line.split(',')
        inname = line_splited[0]
        src_pos_str_idx = inname.find('s=')
        RT_str_idx = inname.find('RT')

        #pdb.set_trace()
        if(src_pos_str_idx > 0):
            src_pos_str = inname[src_pos_str_idx+2:RT_str_idx-1]
            src_pos_vals = src_pos_str.split('_')
            src_pos_vals = [float(p) for p in src_pos_vals]
            check_validity, src_pos_val = check_valid_position(src_pos_vals, src_pos_range, interval_cm)
            if(not check_validity):
                valid_input = False
            else:
                valid_input = True

        # append inname if it is a valid position
        if(valid_input):
            dataset['tarIR'].append(inname.replace('@', ','))
            if(use_ref_IR):
                dataset['refIR'].append(line_splited[2])

        if(len(dataset['tarIR']) > 0 and random.random() < 0.05): # 5% data will be checked for existstence
            tarIR_path = dataset['tarIR'][-1] + '_ch1.npy'
            if (not os.path.exists(tarIR_path)):
                assert (0), tarIR_path + ' not exists'

            if(use_ref_IR):
                refIR_path = dataset['refIR'][-1] + '_ch1.npy'
                if(not os.path.exists(refIR_path)):
                    assert(0), refIR_path + ' not exists'

    manifest.close()

    nSample = len(dataset['tarIR'])
    print('#IR = ' + str(nSample))

    return dataset

class SpecDataset(data.Dataset):
    """
    Audio sample reader to generate spectrogram
    """

    def __init__(self, manifest_path, nMic=8, return_path=False, src_range=None, hop_length=0,
                 interval_cm=1, use_ref_IR=False, start_ratio = 0, end_ratio = 1):
        self.return_path = return_path
        self.use_ref_IR = use_ref_IR

        self.manifest_path = manifest_path

        self.hop_length = hop_length

        #self.nFFT = nFFT

        self.dataset = load_data_list(manifest_path=manifest_path, src_range = src_range, use_ref_IR=use_ref_IR,
                                    interval_cm=interval_cm, start_ratio = start_ratio, end_ratio = end_ratio)

        self.nData = len(self.dataset['tarIR'])
        if(use_ref_IR):
            assert(self.nData == len(self.dataset['refIR']))

        self.nMic = nMic


    def __getitem__(self, idx):

        selected_mics = [1, 2, 3, 4, 5, 6, 7, 8] # fix mic index

        tarIR_list = []

        # load target IR
        for i in range(self.nMic):
            tarIR_path = self.dataset['tarIR'][idx] + '_ch' + str(selected_mics[i]) + '.npy'
            tarIR = np.squeeze(np.load(tarIR_path))
            tarIR_list.append(tarIR)
        tarIR_nMic = torch.FloatTensor(np.stack(tarIR_list)).squeeze()
        if(self.use_ref_IR):
            refIR_list = []
            for i in range(self.nMic):
                refIR_path = self.dataset['refIR'][idx] + '_ch' + str(selected_mics[i]) + '.npy'
                refIR = np.squeeze(np.load(refIR_path))
                refIR_list.append(refIR)
            refIR_nMic = torch.FloatTensor(np.stack(refIR_list)).squeeze()

        return_list = [tarIR_nMic] # time domain signal for now, fft will be used at collate()

        if(self.use_ref_IR):
            return_list.append(refIR_nMic) # time domain signal for now, fft will be used at collate()

        return return_list

    def __len__(self):
        return self.nData

    def list_to_real_tensor(self, inputs):
        N = len(inputs)
        T = max(inp.shape[-1] for inp in inputs)
        if(inputs[0].ndimension() == 2): # multiCH data
            M = inputs[0].shape[0]
            shape = (N, M, T, 2) # 2 = real/imag for torch.fft()
        elif(inputs[0].ndimension() == 1): # singleCH data (clean)
            shape = (M, T, 2) # 2 = real/imag for torch.fft()
        tensor = torch.FloatTensor(*shape).zero_()

        for e, inp in enumerate(inputs):
            tensor[e, ..., :inp.size(-1), 0] = inp # assign data on real part only & keep imag part 0
        return tensor

    def collate(self, inputs):
        input_zips = zip(*inputs)
        
        tarIR_nMic = input_zips.__next__()
        tarIR_batch = self.list_to_real_tensor(tarIR_nMic).cuda()      # cuda before batched fft
        tarH_batch = torch.fft(tarIR_batch, signal_ndim=2) # NxMxT --> NxMxF (T = F)
        #pdb.set_trace()
        tarH_batch = tarH_batch[:, :, :int(tarH_batch.size(2)/2 + 1), :]  # keep only half
        tarH_batch = tarH_batch.unsqueeze(3) # NxMxFx2 --> NxMxFx1x2 # make time dimension for using temporal conv of unet
        if(self.use_ref_IR):
            refIR_nMic = input_zips.__next__()
            refIR_batch = self.list_to_real_tensor(refIR_nMic).cuda()  # cuda before batched fft
            refH_batch = torch.fft(refIR_batch, signal_ndim=2) # NxMxTx2 --> NxMxFx2 (T = F)
            refH_batch = refH_batch[:, :, :int(refH_batch.size(2)/2+1), :] # keep only half
            refH_batch = refH_batch.unsqueeze(3) # NxMxFx2 --> NxMxFx1x2 # make time dimension for using temporal conv of unet

        batch = [tarH_batch]
        if(self.use_ref_IR):
            batch.append(refH_batch)

        return batch
