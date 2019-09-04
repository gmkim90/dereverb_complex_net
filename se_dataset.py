import numpy as np
#import librosa
#import torchaudio
import soundfile as sf # alternative to torchaudio
import torch
from torch.utils import data
import random
from scipy.signal import fftconvolve
import math

import time

#def check_valid_position(idx_to_pos, pos_str_splited, pos_range):
def check_valid_position(pos_val, pos_range, interval_cm=1):
    # pos_range : 3x2 (3 = x,y,z, 2 = min/max)

    #if(pos_range == 'all'): # temporary
#        return True, [0, 0, 0]

    '''
    pos_idx_x = int(pos_str_splited[0])
    pos_idx_y = int(pos_str_splited[1])
    pos_idx_z = int(pos_str_splited[2])

    pos_x = idx_to_pos[0][pos_idx_x]
    pos_y = idx_to_pos[1][pos_idx_y]
    pos_z = idx_to_pos[2][pos_idx_z]
    '''

    pos_x = pos_val[0]
    pos_y = pos_val[1]
    pos_z = pos_val[2]

    #print('pv')
    #print(pos_val)
    #print('pr')
    #print(pos_range)
    #print(' ')

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

import pdb


# Reference
# DATA LOADING - LOAD FILE LISTS
#def load_data_list(folder='./dataset', setname='train'):
#def load_data_list(manifest_path='', pos_val_list=None, pos_range = None):
def load_data_list(manifest_path='', use_localization=False, src_range = None, start_ratio=0.0, end_ratio=1.0, interval_cm=1):

    assert(start_ratio >= 0 and start_ratio <= 1)
    assert(end_ratio >= 0 and end_ratio <= 1)
    assert(start_ratio < end_ratio)

    #assert(setname in ['train', 'val', 'test', 'toy'])

    dataset = {}
    manifest = open(manifest_path, 'r')
    lines = manifest.readlines()
    valid_input = True

    if(manifest_path.find('RT60') >= 0):
        save_RT60 = True
        dataset['RT60'] = []
    else:
        save_RT60 = False

    if (src_range == 'all'):  # special case : all position within room is allowed
        #mic_pos_range = 'all' # ignore posIdx
        #src_pos_range = 'all' # ignore posIdx
        #mic_pos_range = [0, 10000, 0, 10000, 0, 10000] # large enough room
        src_pos_range = [0, 10000, 0, 10000, 0, 10000] # large enough room
    else:
        #mic_pos_range = pos_range[:6] # will not be used anymore
        #src_pos_range = pos_range[6:]
        src_pos_range = src_range


    print("Loading files from manifst " + manifest_path)
    dataset['innames'] = []
    dataset['outnames'] = []
    dataset['src_pos'] = []

    nLine = len(lines)
    startIdx = math.floor(start_ratio*nLine)
    endIdx = math.floor(end_ratio*nLine)

    #for line in lines:
    for i in range(startIdx, endIdx):
        line = lines[i]
        line_repr = line.replace('\n', '')
        line_splited = line_repr.split(',')
        inname = line_splited[0]
        #posIdx = int(line_splited[2].replace('\n', ''))
        src_pos_str_idx = inname.find('s=')
        RT_str_idx = inname.find('RT')

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
            dataset['innames'].append(inname.replace('@', ','))
            dataset['outnames'].append(line_splited[1])
            if(save_RT60):
                dataset['RT60'].append(line_splited[2])
            if(use_localization):
                dataset['src_pos'].append(src_pos_val)

    manifest.close()
    nSample = len(dataset['innames'])
    print('#sample = ' + str(nSample))

    return dataset

class SpecDataset(data.Dataset):
    """
    Audio sample reader to generate spectrogram
    """

    def __init__(self, manifest_path, stft, nMic=8, sampling_method='no', subset1=None, subset2=None, return_path=False, fix_len_by_cl='input',
                 load_IR=False, use_localization=False, src_range=None, nSource=1, start_ratio=0.0, end_ratio=1.0,
                 clamp_frame=0, ref_mic_direct_td_subtract=True, interval_cm=1):
        self.return_path = return_path

        #self.clamp_src = clamp_src
        self.clamp_frame = clamp_frame
        self.ref_mic_direct_td_subtract = ref_mic_direct_td_subtract

        # ver 1. all of wav data is loaded in advance
        #dataset = load_data_list(manifest_path=manifest_path)
        #self.dataset = load_data(dataset)

        # ver2. load wav file at every iteration
        self.dataset = load_data_list(manifest_path=manifest_path, use_localization=use_localization, src_range = src_range,
                                      start_ratio=start_ratio, end_ratio=end_ratio, interval_cm=interval_cm)
        if('RT60' in self.dataset):
            self.return_RT60 = True
            self.return_path = True
        else:
            self.return_RT60 = False

        if(len(self.dataset['src_pos']) > 0):
            self.return_src_pos = True
        else:
            self.return_src_pos = False

        #self.input_data_as_IR = input_data_as_IR # normally, reverberation is modeled as convolution in time domain
                                                 # instead of multiplication in freq domain

        self.file_names = self.dataset['innames']
        self.nData = len(self.dataset['outnames'])
        self.nSource = nSource # how many sources to use

        self.load_IR = load_IR

        self.stft = stft

        self.fix_len_by_cl = fix_len_by_cl

        self.nMic = nMic
        self.sampling_method = sampling_method
        self.subset1 = subset1
        self.subset2 = subset2

    def __getitem__(self, idx):
        # ver1. all of wav data is loaded in advance
        #mixed = torch.from_numpy(self.dataset['inaudio'][idx]).type(torch.FloatTensor)
        #clean = torch.from_numpy(self.dataset['outaudio'][idx]).type(torch.FloatTensor)

        # ver2. load wav file at every iteration

        # ver3. consider mic sampling method
        if(self.sampling_method == 'no'):
            selected_mics = [1, 2, 3, 4, 5, 6, 7, 8] # fix mic index
        elif(self.sampling_method == 'random'):
            selected_mics = random.sample(set([1, 2, 3 ,4, 5, 6, 7, 8]), self.nMic)
        elif(self.sampling_method == 'ref_random'):
            selected_mics_wo_ref = random.sample(set([2, 3, 4, 5, 6, 7, 8]), self.nMic-1)
            selected_mics = [1] + selected_mics_wo_ref
        elif(self.sampling_method == 'ref_manual'):
            p = random.uniform(0, 1)
            if(p > 0.5):
                selected_mics = self.subset1
            else:
                selected_mics = self.subset2
        inputData = []

        if(self.nSource == 1):
            outputData, sr = sf.read(self.dataset['outnames'][idx])
            # if (self.clamp_src > 0):
            #     outputData = outputData[self.clamp_src:-self.clamp_src]  # remove front/end samples (i.e., remove front/end silence)
        elif(self.nSource > 1):
            outputData_list = []
            for n in range(self.nSource):
                idx_n = (idx + n)%self.nData
                outputData_n, sr = sf.read(self.dataset['outnames'][idx_n])
                # if(self.clamp_src > 0):
                #     outputData_n = outputData_n[self.clamp_src:-self.clamp_src] # remove front/end samples (i.e., remove front/end silence)
                outputData_list.append(outputData_n)
                del outputData_n
            outputData = np.hstack(outputData_list)
            del outputData_list


        if(self.load_IR):
            for i in range(self.nMic):
                load_path = self.dataset['innames'][idx] + '_ch' + str(selected_mics[i]) + '.npy'
                IR = np.squeeze(np.load(load_path))
                inputData_single = fftconvolve(outputData, IR)
                if(self.ref_mic_direct_td_subtract):
                    if(i == 0):
                        tau1 = IR.argmax()
                    inputData_single = inputData_single[tau1:]
                inputData.append(inputData_single)
                del inputData_single, IR
        else:
            for i in range(self.nMic):
                load_path = self.dataset['innames'][idx] + '_ch' + str(selected_mics[i]) + '.wav'
                inputData_single, sr = sf.read(load_path)
                inputData.append(inputData_single)
                del inputData_single

        mixed = torch.FloatTensor(np.stack(inputData)).squeeze()
        clean = torch.FloatTensor(outputData)
        del inputData, outputData

        T = clean.nelement()
        if(self.fix_len_by_cl == 'input'):
            mixed = mixed[:, :T]

        mixedSTFT = self.stft(mixed.cuda())
        cleanSTFT = self.stft(clean.cuda())

        if(self.clamp_frame > 0):
            # print('CLAMP before')
            # print('mixedSTFT')
            # print(mixedSTFT.size())
            # print('cleanSTFT')
            # print(cleanSTFT.size())
            mixedSTFT = mixedSTFT[:, :, self.clamp_frame:-self.clamp_frame, :] # MxFxTx2
            cleanSTFT = cleanSTFT[:, self.clamp_frame:-self.clamp_frame, :]  # MxFxTx2
            # print('CLAMP after')
            # print('mixedSTFT')
            # print(mixedSTFT.size())
            # print('cleanSTFT')
            # print(cleanSTFT.size())

        #return_list = [mixedSTFT, cleanSTFT, mixed, clean]
        return_list = [mixedSTFT, cleanSTFT] # do not use time domain signal anymore
        del mixed, clean

        if(self.return_path):
            return_list.append(self.dataset['innames'][idx])

        if(self.return_RT60):
            return_list.append(self.dataset['RT60'][idx])

        if(self.return_src_pos):
            return_list.append(self.dataset['src_pos'][idx])

        return return_list

        '''
        if(not self.return_path): # train.py
            if(not self.return_RT60):
                return mixedSTFT, cleanSTFT, mixed, clean
            else:
                RT60 = self.dataset['RT60'][idx]
                return mixedSTFT, cleanSTFT, mixed, clean, RT60
        else: # test.py
            if(not self.return_RT60):
                return mixedSTFT, cleanSTFT, self.dataset['innames'][idx], mixed, clean
            else:
                RT60 = self.dataset['RT60'][idx]
                return mixedSTFT, cleanSTFT, self.dataset['innames'][idx], mixed, clean, RT60
        '''


    def __len__(self):
        return len(self.file_names)

    def zero_pad_concat_time(self, inputs):
        bsz = len(inputs)
        max_t = max(inp.shape[-1] for inp in inputs)
        if(inputs[0].ndimension() == 2): # multiCH data
            nCH = inputs[0].shape[0]
            shape = (bsz, nCH, max_t)
        elif(inputs[0].ndimension() == 1): # singleCH data (clean)
            shape = (bsz, max_t)
        #input_mat = np.zeros(shape, dtype=np.float32)
        input_mat = torch.FloatTensor(*shape).zero_()

        for e, inp in enumerate(inputs):
            input_mat[e, ..., :inp.size(-1)] = inp
        return input_mat

    def zero_pad_concat_STFT(self, inputs):
        bsz = len(inputs)
        max_t = max(inp.shape[-2] for inp in inputs)
        if(inputs[0].ndimension() == 4): # multiCH data
            nCH = inputs[0].shape[0]
            F = inputs[0].shape[1]
            shape = (bsz, nCH, F, max_t, 2)
        elif(inputs[0].ndimension() == 3): # singleCH data (clean)
            F = inputs[0].shape[0]
            shape = (bsz, F, max_t, 2)
        input_mat = torch.FloatTensor(*shape).zero_()

        for e, inp in enumerate(inputs):
            input_mat[e, ..., :inp.size(-2), :] = inp

        del inputs

        return input_mat

    def collate(self, inputs):
        # ver 1
        '''
        if(not self.return_path):
            if(not self.return_RT60):
                mixeds_STFT, cleans_STFT, mixeds_time, cleans_time = zip(*inputs)
            else:
                mixeds_STFT, cleans_STFT, mixeds_time, cleans_time, RT60 = zip(*inputs)
                RT60s = torch.IntTensor([int(i) for i in RT60])
        else:
            if(not self.return_RT60):
                mixeds_STFT, cleans_STFT, reverb_paths, mixeds_time, cleans_time = zip(*inputs)
            else:
                mixeds_STFT, cleans_STFT, reverb_paths, mixeds_time, cleans_time, RT60 = zip(*inputs)
                RT60s = torch.IntTensor([int(i) for i in RT60])
        '''

        # ver 2
        #pdb.set_trace()
        #nInput = len(inputs[0]) # debug purpose
        #print('#input = ' + str(nInput)) # debug purpose

        input_zips = zip(*inputs)
        
        mixeds_STFT = input_zips.__next__()
        cleans_STFT = input_zips.__next__()
        #mixeds_time = input_zips.__next__() # do not use time domain signal anymore
        #cleans_time = input_zips.__next__() # do not use time domain signal anymore        
        if(self.return_path):
            reverb_paths = input_zips.__next__()
        if(self.return_RT60):
            RT60 = input_zips.__next__()
            RT60s = torch.IntTensor([int(i) for i in RT60])
        if(self.return_src_pos):
            src_pos = input_zips.__next__()        
        del input_zips

        # ver3. method without __next__
        #mixeds_STFT, cleans_STFT = zip(*inputs)

        #seq_lens_STFT = torch.IntTensor([i.shape[-2] for i in mixeds_STFT])
        #seq_lens_time = torch.IntTensor([i.shape[-1] for i in mixeds_time])

        seq_lens_STFT = torch.IntTensor([i.shape[-2] for i in cleans_STFT]) # measured by clean
        #seq_lens_time = torch.IntTensor([i.shape[-1] for i in cleans_time]) # measured by clean

        x_STFT = torch.FloatTensor(self.zero_pad_concat_STFT(mixeds_STFT))
        y_STFT = torch.FloatTensor(self.zero_pad_concat_STFT(cleans_STFT)).squeeze() # may contain garbage dimension

        del cleans_STFT, mixeds_STFT
        #x_time = torch.FloatTensor(self.zero_pad_concat_time(mixeds_time))
        #y_time = torch.FloatTensor(self.zero_pad_concat_time(cleans_time))

        #pdb.set_trace()

        #batch = [x_STFT, y_STFT, seq_lens_STFT, x_time, y_time, seq_lens_time]
        if(y_STFT.size(0) > 100): # possibly, sample dimension is missing (cuz minibatch size = 1)
            #x_STFT = x_STFT.unsqueeze(0) # x is already unsqueezed in the front part
            y_STFT = y_STFT.unsqueeze(0)
        batch = [x_STFT, y_STFT, seq_lens_STFT]
        #pdb.set_trace()
        if(self.return_path):
            batch.append(reverb_paths)

        if(self.return_RT60):
            batch.append(RT60s)

        if(self.return_src_pos):
            batch.append(src_pos)

        # ver 1
        '''
        if (not self.return_path):
            if(not self.return_RT60):
                batch = [x_STFT, y_STFT, seq_lens_STFT, x_time, y_time, seq_lens_time]
            else:
                batch = [x_STFT, y_STFT, seq_lens_STFT, x_time, y_time, seq_lens_time, RT60s]
        else:
            if(not self.return_RT60):
                batch = [x_STFT, y_STFT, seq_lens_STFT, x_time, y_time, seq_lens_time, reverb_paths]
            else:
                batch = [x_STFT, y_STFT, seq_lens_STFT, x_time, y_time, seq_lens_time, reverb_paths, RT60s]
        '''

        return batch
