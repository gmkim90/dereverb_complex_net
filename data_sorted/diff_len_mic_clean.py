import soundfile as sf
import scipy.io as sio
from tqdm import trange
import numpy as np

# tr + dt + et
'''
manifest_list = ['reverb_tr_simu_8ch_paired.csv', 'reverb_dt_simu_8ch_paired.csv', 'reverb_et_simu_8ch_paired.csv']
nSample = [7861, 1484, 2176]

len_info_tr = np.zeros((nSample[0], 3)) # mixed, clean, mixed-clean
len_info_dt = np.zeros((nSample[1], 3)) # mixed, clean, mixed-clean
len_info_et = np.zeros((nSample[2], 3)) # mixed, clean, mixed-clean
len_info_list = [len_info_tr, len_info_dt, len_info_et]
'''

# tr + dt

manifest_list = ['reverb_tr_simu_8ch_paired.csv', 'reverb_dt_simu_8ch_paired.csv']
nSample = [7861, 1484]

len_info_tr = np.zeros((nSample[0], 3)) # mixed, clean, mixed-clean
len_info_dt = np.zeros((nSample[1], 3)) # mixed, clean, mixed-clean
len_info_list = [len_info_tr, len_info_dt]

# dt + et
#manifest_list = ['reverb_dt_simu_8ch_paired.csv', 'reverb_et_simu_8ch_paired.csv']
#nSample = [1484, 2176]

#len_info_dt = np.zeros((nSample[0], 3)) # mixed, clean, mixed-clean
#len_info_et = np.zeros((nSample[1], 3)) # mixed, clean, mixed-clean
#len_info_list = [len_info_dt, len_info_et]

# tr only
#manifest_list = ['reverb_tr_simu_8ch_paired.csv']
#nSample = [7861]
#len_info_tr = np.zeros((nSample[0], 3)) # mixed, clean, mixed-clean
#len_info_list = [len_info_tr]

for i in range(len(manifest_list)):
    print('analyze ' + manifest_list[i])
    #print 'analyze ' + manifest_list[i]
    manifest = open(manifest_list[i], 'r')
    lines = manifest.readlines()

    #for line in lines:
    for j in trange(0, len(lines)):
        line = lines[j]
        line = line.replace('\n', '')
        line_splited = line.split(',')
        path_reverb, path_clean = line_splited[0], line_splited[1]

        reverb_ch1, fs = sf.read(path_reverb + '_ch1.wav')
        clean, fs = sf.read(path_clean)

        len_info_list[i][j][0] = len(reverb_ch1)
        len_info_list[i][j][1] = len(clean)
        len_info_list[i][j][2] = len_info_list[i][j][0] - len_info_list[i][j][1]

        #print(len_info_list[i][j][0])
        #print(len_info_list[i][j][1])
        #print(len_info_list[i][j][2])

    manifest.close()

#sio.savemat('len_info_reverb.mat', {'len_info_tr':len_info_list[0], 'len_info_dt':len_info_list[1], 'len_info_et':len_info_list[2]})
#sio.savemat('len_info_reverb.mat', {'len_info_dt':len_info_list[0], 'len_info_et':len_info_list[1]})

sio.savemat('len_info_reverb_trdt.mat', {'len_info_tr':len_info_list[0], 'len_info_dt':len_info_list[1]})
