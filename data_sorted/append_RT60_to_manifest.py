from tqdm import trange
import numpy as np
import pdb

# process csv to dictionary
meta_name = ['meta_info_tr.csv', 'meta_info_dt.csv']
meta_list_all = []
for i in range(len(meta_name)):
    print('process ' + meta_name[i] + ' to dictionary')
    fp = open(meta_name[i], 'r')
    lines = fp.readlines()

    meta_list = []
    for j in trange(0, len(lines)):
        line = lines[j]
        line = line.replace('\n', '')
        line_splited = line.split(',')
        meta = {'reverb': line_splited[0], 'RIR': line_splited[1], 'RT60': line_splited[2]}
        meta_list.append(meta)
    meta_list_all.append(meta_list)

    fp.close()

np.save('meta_list_all.npy', meta_list_all)

# write new manifest
#meta_list_all = np.load('meta_list_all.npy')
r_list = ['reverb_tr_simu_8ch_paired.csv', 'reverb_dt_simu_8ch_paired.csv']
w_list = ['reverb_tr_simu_8ch_RT60.csv', 'reverb_dt_simu_8ch_RT60.csv']

for i in range(len(r_list)):
    fp_r = open(r_list[i], 'r')
    fp_w = open(w_list[i], 'w')
    print('append RT60 to ' + r_list[i])

    meta_list = meta_list_all[i]
    lines = fp_r.readlines()
    for j in trange(0, len(lines)):
        line = lines[j]
        line = line.replace('\n', '')
        line_splited = line.split(',')
        clean_path = line_splited[1]
        ID = clean_path.split('/')[-1].split('.')[0]

        match_dict = next(item for item in meta_list if item['reverb'].find(ID) >= 0)
        RT60 = match_dict['RT60']

        line_w = line_splited[0] + ',' + line_splited[1] + ',' + RT60
        fp_w.write(line_w + '\n')