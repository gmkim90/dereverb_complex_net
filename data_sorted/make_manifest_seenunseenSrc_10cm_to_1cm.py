from tqdm import trange
import os
import numpy as np

manifest_r_list = ['L553_fixedmic_hyper_1cm_RT0.2_tr.csv', 'L553_fixedmic_hyper_1cm_RT0.2_tr.csv']
manifest_w_list = ['L553_fixedmic_hyper_1cm_seenSrc_10cm_to_1cm.csv', 'L553_fixedmic_hyper_1cm_unseenSrc_10cm_to_1cm.csv']
src_dir_list = ['/home/kenkim/librispeech_kaldi/LibriSpeech/train/wav', '/home/kenkim/librispeech_kaldi/LibriSpeech/test_clean/wav']

seen_src_list = ['6078-54013-0049.wav', '1336-138113-0000.wav', '6227-36632-0080.wav', '6963-81511-0050.wav', '5036-18451-0000.wav']
unseen_src_list = ['5105-28241-0010.wav', '1995-1826-0025.wav', '4970-29095-0015.wav', '6829-68769-0044.wav', '3570-5695-0010.wav']

src_biglist = [seen_src_list, unseen_src_list]

#nSource = 5
nSource = len(seen_src_list)
assert(nSource == len(unseen_src_list))

src_range_list = [4.3, 4.6, 4.3, 4.6, 1.3, 1.3]

def is_valid_position(pos_range, value):
    # pos_range: 1x6 (xmin, xmax, ymin, ymax, zmin, zmax)
    # value: 1x3 (x,y,z)

    x = value[0]
    y = value[1]
    z = value[2]

    xmin = pos_range[0]
    xmax = pos_range[1]
    ymin = pos_range[2]
    ymax = pos_range[3]
    zmin = pos_range[4]
    zmax = pos_range[5]

    if(x >= xmin and x <= xmax and
    y >= ymin and y <= ymax and
    z >= zmin and z <= zmax):
        return True

    return False


for i in range(len(manifest_r_list)):
    manifest_r = open(manifest_r_list[i], 'r')
    manifest_w = open(manifest_w_list[i], 'w')
    print(manifest_r_list[i] + ' --> ' + manifest_w_list[i])

    lines = manifest_r.readlines()
    print('get unique IR list with restricted range')
    IR_unique_list = []
    for j in range(len(lines)):
        line = lines[j]
        line_splited = line.split(',')
        IR_path = line_splited[0]
        sIdx = IR_path.find('s=')
        RTIdx = IR_path.find('_RT')
        pos_str = IR_path[sIdx+2:RTIdx]
        pos_str_splited = pos_str.split('_')
        pos_x = float(pos_str_splited[0])
        pos_y = float(pos_str_splited[1])
        pos_z = float(pos_str_splited[2])

        valid_position = is_valid_position(src_range_list, [pos_x, pos_y, pos_z])

        if(valid_position and IR_path not in IR_unique_list):
            IR_unique_list.append(IR_path)


    src_dir = src_dir_list[i]
    src_list = src_biglist[i]

    for n in trange(0, nSource):
        src_path = src_dir + '/' + src_list[n]
        print('write with src ' + src_path)
        for j in range(len(IR_unique_list)):
            IR_path = IR_unique_list[j]
            line_w = IR_path + ',' + src_path
            manifest_w.write(line_w + '\n')

    manifest_r.close()
    manifest_w.close()