import numpy as np
#import librosa
#import torchaudio
import soundfile as sf # alternative to torchaudio
import torch
from torch.utils import data
import random
import time

import pdb

# Reference
# DATA LOADING - LOAD FILE LISTS
#def load_data_list(folder='./dataset', setname='train'):
def load_data_list(manifest_path=''):

    #assert(setname in ['train', 'val', 'test', 'toy'])

    dataset = {}
    manifest = open(manifest_path, 'r')
    lines = manifest.readlines()

    if(manifest_path.find('RT60') >= 0):
        save_RT60 = True
        dataset['RT60'] = []
    else:
        save_RT60 = False

    print("Loading files from manifest " + manifest_path)
    dataset['innames'] = []
    dataset['outnames'] = []
    for line in lines:
        line = line.replace('\n', '')
        line_splited = line.split(',')
        dataset['innames'].append(line_splited[0])
        dataset['outnames'].append(line_splited[1])
        if(save_RT60):
            dataset['RT60'].append(line_splited[2])

    manifest.close()

    return dataset

class SpecDataset(data.Dataset):
    """
    Audio sample reader to generate spectrogram
    """

    def __init__(self, manifest_path, stft, nMic=8, sampling_method='no', subset1=None, subset2=None, return_path=False, fix_len_by_cl='input'):
        self.return_path = return_path

        # ver 1. all of wav data is loaded in advance
        #dataset = load_data_list(manifest_path=manifest_path)
        #self.dataset = load_data(dataset)

        # ver2. load wav file at every iteration
        self.dataset = load_data_list(manifest_path=manifest_path)
        if('RT60' in self.dataset):
            self.return_RT60 = True
            self.return_path = True
        else:
            self.return_RT60 = False
        self.file_names = self.dataset['innames']

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
        # librosa & numpy
        #inputData_single, sr = librosa.load(self.dataset['innames'][idx] + '_ch' + str(selected_mics[0]) + '.wav', sr=None)
        #inputData = np.expand_dims(inputData_single, axis=0)

        inputData = []

        # torchaudio & torch
        '''
        for i in range(self.nMic):
            inputData_single, sr = torchaudio.load(self.dataset['innames'][idx] + '_ch' + str(selected_mics[i]) + '.wav')
            inputData.append(inputData_single)
        inputData = torch.stack(inputData)
        '''

        # soundfile
        for i in range(self.nMic):
            inputData_single, sr = sf.read(self.dataset['innames'][idx] + '_ch' + str(selected_mics[i]) + '.wav')
            inputData.append(inputData_single)
        inputData = torch.FloatTensor(np.stack(inputData))

        # soundfile
        outputData, sr = sf.read(self.dataset['outnames'][idx])
        outputData = torch.FloatTensor(outputData)

        # soundfile
        mixed = inputData.squeeze()
        clean = outputData

        #T = clean.size(0)
        T = clean.nelement()
        if(self.fix_len_by_cl == 'input'):
            mixed = mixed[:, :T]

        mixedSTFT = self.stft(mixed.cuda())
        cleanSTFT = self.stft(clean.cuda())

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
        return input_mat

    def collate(self, inputs):
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

        #seq_lens_STFT = torch.IntTensor([i.shape[-2] for i in mixeds_STFT])
        #seq_lens_time = torch.IntTensor([i.shape[-1] for i in mixeds_time])

        seq_lens_STFT = torch.IntTensor([i.shape[-2] for i in cleans_STFT]) # measured by clean
        seq_lens_time = torch.IntTensor([i.shape[-1] for i in cleans_time]) # measured by clean


        #print(mixeds_STFT[0].size())
        #print(mixeds_time[0].size())
        #print(seq_lens_STFT)
        #print(seq_lens_time)

        x_STFT = torch.FloatTensor(self.zero_pad_concat_STFT(mixeds_STFT))
        y_STFT = torch.FloatTensor(self.zero_pad_concat_STFT(cleans_STFT)).squeeze() # may contain garbage dimension

        x_time = torch.FloatTensor(self.zero_pad_concat_time(mixeds_time))
        y_time = torch.FloatTensor(self.zero_pad_concat_time(cleans_time))

        #pdb.set_trace()

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

        return batch