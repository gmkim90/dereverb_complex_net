import numpy as np

import soundfile as sf # alternative to torchaudio
import torch
from torch.utils import data
import random
import math
import os
from scipy.signal import fftconvolve

import time
import pdb

def get_nFrame_from_nTime(nTime_list, win_size, hop_size):
    N = len(nTime_list)
    nFrame_tensor = torch.IntTensor(N)
    for n in range(N):
        nFrame_tensor[n] = math.ceil((nTime_list[n]-(win_size-1)-1)/hop_size) + 2
        #nFrame_tensor[n] = math.ceil((nTime_list[n]-(win_size-1)-1)/hop_size) + 1 #  on pytorch forum
        #nFrame_tensor[n] = math.ceil((nTime_list[n] - (win_size - 1) - 1) / hop_size)

    return nFrame_tensor


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

def load_data_list_src(manifest_path='', src_range = None, use_ref_IR=False,
                   start_ratio=0.0, end_ratio=1.0, interval_cm=1):

    assert(start_ratio >= 0 and start_ratio <= 1)
    assert(end_ratio >= 0 and end_ratio <= 1)
    assert(start_ratio < end_ratio)

    dataset = {}
    manifest = open(manifest_path, 'r')
    lines = manifest.readlines()
    valid_input = True

    if (src_range == 'all'):  # special case : all position within room is allowed
        src_pos_range = [0, 10000, 0, 10000, 0, 10000] # large enough room
    else:
        src_pos_range = src_range


    print("Loading files from manifst " + manifest_path)
    dataset['mic'] = []
    dataset['clean'] = []
    if(use_ref_IR):
        dataset['ref_IR'] = []

    nLine = len(lines)
    startIdx = math.floor(start_ratio*nLine)
    endIdx = math.floor(end_ratio*nLine)

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
            dataset['mic'].append(inname.replace('@', ','))
            dataset['clean'].append(line_splited[1])
            if(use_ref_IR):
                dataset['ref_IR'].append(line_splited[2])


        if(len(dataset['mic']) > 0 and random.random() < 0.05): # 5% data will be checked for existstence
            IR_path = dataset['mic'][-1] + '_ch1.npy'
            src_path = dataset['clean'][-1]
            if(not os.path.exists(IR_path)):
                assert(0), IR_path + ' (target position) not exists'
            if(not os.path.exists(src_path)):
                assert(0), src_path + ' clean speech not exists'
            if(use_ref_IR):
                refIR_path = dataset['ref_IR'][-1] + '_ch1.npy'
                if(not os.path.exists(refIR_path)):
                    assert(0), refIR_path + ' not exists'

    manifest.close()

    nSample = len(dataset['mic'])
    print('#sample = ' + str(nSample))

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

        if(self.return_path):
            return_list.append(self.dataset['tarIR'][idx])

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
            shape = (N, T, 2) # 2 = real/imag for torch.fft()
        tensor = torch.FloatTensor(*shape).zero_()

        for e, inp in enumerate(inputs):
            tensor[e, ..., :inp.size(-1), 0] = inp # assign data on real part only & keep imag part 0
        return tensor

    def collate(self, inputs):
        input_zips = zip(*inputs)
        
        tarIR_nMic = input_zips.__next__()
        N = len(tarIR_nMic)
        tarIR_batch = self.list_to_real_tensor(tarIR_nMic).cuda()      # cuda before batched fft
        #tarH_batch = torch.fft(tarIR_batch, signal_ndim=2) # NxMxTx2 --> NxMxFx2 (T = F) --> 2D fft

        tarIR_batch = tarIR_batch.view(tarIR_batch.size(0)*tarIR_batch.size(1), tarIR_batch.size(2), tarIR_batch.size(3))
        tarH_batch = torch.fft(tarIR_batch, signal_ndim=1) # NMxTx2 --> NMxFx2 (T = F) --> 1D fft
        tarH_batch = tarH_batch.view(N, -1, tarH_batch.size(1), tarH_batch.size(2))
        #pdb.set_trace()
        tarH_batch = tarH_batch[:, :, :int(tarH_batch.size(2)/2 + 1), :]  # keep only half
        tarH_batch = tarH_batch.unsqueeze(3) # NxMxFx2 --> NxMxFx1x2 # make time dimension for using temporal conv of unet
        if(self.use_ref_IR):
            refIR_nMic = input_zips.__next__()
            refIR_batch = self.list_to_real_tensor(refIR_nMic).cuda()  # cuda before batched fft
            #refH_batch = torch.fft(refIR_batch, signal_ndim=2) # NxMxTx2 --> NxMxFx2 (T = F) --> 2D fft
            refIR_batch = refIR_batch.view(refIR_batch.size(0)*refIR_batch.size(1), refIR_batch.size(2), refIR_batch.size(3))
            refH_batch = torch.fft(refIR_batch, signal_ndim=1) # NxMxTx2 --> NxMxFx2 (T = F) --> 1D fft
            refH_batch = refH_batch.view(N, -1, refH_batch.size(1), refH_batch.size(2))
            refH_batch = refH_batch[:, :, :int(refH_batch.size(2)/2+1), :] # keep only half
            refH_batch = refH_batch.unsqueeze(3) # NxMxFx2 --> NxMxFx1x2 # make time dimension for using temporal conv of unet

        batch = [tarH_batch]
        if(self.use_ref_IR):
            batch.append(refH_batch)

        if(self.return_path):
            reverb_path = input_zips.__next__()
            batch.append(reverb_path)

        return batch


class SpecDataset_src(data.Dataset): # src-dependent

    def __init__(self, manifest_path, stft, win_size, hop_size, nMic=2, return_path=False,
                 src_range=None, nSource=1, start_ratio=0.0, end_ratio=1.0,
                 interval_cm=1, use_ref_IR=False):

        self.win_size = win_size
        self.hop_size = hop_size

        self.return_path = return_path

        self.manifest_path = manifest_path

        self.dataset = load_data_list_src(manifest_path=manifest_path, src_range=src_range,
                                      start_ratio=start_ratio, end_ratio=end_ratio, interval_cm=interval_cm,
                                      use_ref_IR=use_ref_IR)

        self.file_names = self.dataset['mic']
        self.nData = len(self.dataset['clean'])
        self.nSource = nSource  # how many sources to use

        self.use_ref_IR = use_ref_IR

        self.stft = stft

        self.nMic = nMic

    def __getitem__(self, idx):
        inputData = []
        outputData, sr = sf.read(self.dataset['clean'][idx])
        T = outputData.shape[0]

        #if (self.do_1st_frame_clamp):
        outputData = np.pad(outputData, (self.hop_size, 0), 'constant')

        # load IR
        for i in range(self.nMic):
            load_path = self.dataset['mic'][idx] + '_ch' + str(i+1) + '.npy'
            IR = np.squeeze(np.load(load_path))
            inputData_single = fftconvolve(outputData, IR)
            if (i == 0):
                tau1 = IR.argmax()
            inputData_single = inputData_single[tau1:]
            inputData.append(inputData_single)
            del inputData_single, IR

        if (self.use_ref_IR):
            refmic_data = []
            for i in range(self.nMic):
                load_path = self.dataset['ref_IR'][idx] + '_ch' + str(i+1) + '.npy'
                IR = np.squeeze(np.load(load_path))
                refmic_single = fftconvolve(outputData, IR)
                if (i == 0):
                    tau1 = IR.argmax()
                refmic_single = refmic_single[tau1:]
                refmic_data.append(refmic_single)
                del refmic_single, IR
            refmic = torch.FloatTensor(np.stack(refmic_data)).squeeze()
            del refmic_data

        mic = torch.FloatTensor(np.stack(inputData)).squeeze()
        clean = torch.FloatTensor(outputData)
        del inputData, outputData

        # torch.save([mixed, clean], 'mixed.pth')


        #if (self.fix_len_by_cl == 'input'):
        mic = mic[:, :T+self.hop_size] # for clamp frame
        if (self.use_ref_IR):
            refmic = refmic[:, :T+self.hop_size] # for clamp frame

        return_list = [mic, clean, T]

        if (self.use_ref_IR):
            return_list.append(refmic)

        if (self.return_path):
            return_list.append(self.dataset['mic'][idx])

        return return_list

    def __len__(self):
        return len(self.file_names)

    def zero_pad_concat_time(self, inputs):
        bsz = len(inputs)
        max_t = max(inp.shape[-1] for inp in inputs)
        if (inputs[0].ndimension() == 2):  # multiCH data
            nCH = inputs[0].shape[0]
            shape = (bsz, nCH, max_t)
        elif (inputs[0].ndimension() == 1):  # singleCH data (clean)
            shape = (bsz, max_t)
        # input_mat = np.zeros(shape, dtype=np.float32)
        input_mat = torch.FloatTensor(*shape).zero_()

        for e, inp in enumerate(inputs):
            input_mat[e, ..., :inp.size(-1)] = inp
        return input_mat

    def collate(self, inputs):
        input_zips = zip(*inputs)

        mics_list = input_zips.__next__()
        cleans_list = input_zips.__next__()
        nTime_list = input_zips.__next__()
        if (self.use_ref_IR):
            refmics_list = input_zips.__next__()
        if (self.return_path):
            reverb_paths = input_zips.__next__()

        # Make list to batched torch.FloatTensor()
        mics_batch = self.zero_pad_concat_time(mics_list)
        cleans_batch = self.zero_pad_concat_time(cleans_list)
        N, M = mics_batch.size(0), mics_batch.size(1)

        # batch version stft
        clean_STFT = self.stft(cleans_batch.cuda())
        mic_STFT = self.stft(mics_batch.view(-1, mics_batch.size(2)).cuda())
        mic_STFT = mic_STFT.view(N, M, mic_STFT.size(1), mic_STFT.size(2), -1) # NxMxFxTx2

        # if (self.do_1st_frame_clamp):
        mic_STFT = mic_STFT[:, :, :, 1:, :]  # NxMxFxTx2
        clean_STFT = clean_STFT[:, :, 1:, :]  # NxFxTx2
        if (self.use_ref_IR):
            refmics_batch = self.zero_pad_concat_time(refmics_list)
            refmic_STFT = self.stft(refmics_batch.view(-1, refmics_batch.size(2)).cuda())
            refmic_STFT = refmic_STFT.view(N, M, refmic_STFT.size(1), refmic_STFT.size(2), -1) # NxMxFxTx2
            refmic_STFT = refmic_STFT[:, :, :, 1:, :]  # NxMxFxTx2

        nFrame_tensor = get_nFrame_from_nTime(nTime_list, self.win_size, self.hop_size)
        #assert(mic_STFT.size(3) == nFrame_tensor.max().item()), 'batch STFT max length = ' + str(mic_STFT.size(3)) + ' VS. nFrame_tensor.max() = ' + str(nFrame_tensor.max().item())


        #if (y_STFT.size(0) > 100):  # possibly, sample dimension is missing (cuz minibatch size = 1)
            #y_STFT = y_STFT.unsqueeze(0)

        batch = [mic_STFT, clean_STFT, nFrame_tensor] # note that seq_lens_STFT all have same value
        if (self.use_ref_IR):
            batch.append(refmic_STFT)

        if (self.return_path):
            batch.append(reverb_paths)

        return batch
