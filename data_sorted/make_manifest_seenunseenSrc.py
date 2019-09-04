from tqdm import trange
import os
import numpy as np

manifest_r_list = ['L553_fixedmic_hyper_0.1cm_RT0.2_nongrid_0.1.csv', 'L553_fixedmic_hyper_0.1cm_RT0.2_nongrid_0.1.csv']
manifest_w_list = ['L553_fixedmic_hyper_0.1cm_seenSrc.csv', 'L553_fixedmic_hyper_0.1cm_unseenSrc.csv']
src_dir_list = ['/home/kenkim/librispeech_kaldi/LibriSpeech/train/wav', '/home/kenkim/librispeech_kaldi/LibriSpeech/test_clean/wav']

nSource = 5

for i in range(len(manifest_r_list)):
    manifest_r = open(manifest_r_list[i], 'r')
    manifest_w = open(manifest_w_list[i], 'w')
    print(manifest_r_list[i] + ' --> ' + manifest_w_list[i])

    lines = manifest_r.readlines()
    print('get unique IR list')
    IR_unique_list = []
    for j in range(len(lines)):
        line = lines[j]
        line_splited = line.split(',')
        IR_path = line_splited[0]
        if(IR_path not in IR_unique_list):
            IR_unique_list.append(IR_path)

    src_dir = src_dir_list[i]
    files = os.listdir(src_dir)
    #randpermIdx = random.randint(0, len(files)-1)
    randpermIdx = np.random.permutation(len(files))
    for n in trange(0, nSource):
        src_path = src_dir + '/' + files[randpermIdx[n]]
        print('write with src ' + src_path)
        for j in range(len(IR_unique_list)):
            IR_path = IR_unique_list[j]
            line_w = IR_path + ',' + src_path
            manifest_w.write(line_w + '\n')

    manifest_r.close()
    manifest_w.close()