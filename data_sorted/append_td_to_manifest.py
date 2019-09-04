import scipy.io as sio
from tqdm import trange
import pdb

# dt & et
'''
r_list = ['reverb_dt_simu_8ch_paired.csv', 'reverb_et_hard6cases_simu_8ch_paired.csv']
w1_list = ['reverb_dt_simu_8ch_td_gt.csv', 'reverb_et_hard6cases_simu_8ch_td_gt.csv']
w2_list = ['reverb_dt_simu_8ch_td_est.csv', 'reverb_et_hard6cases_simu_8ch_td_est.csv']
scp_list = ['rev_wav_dt.txt', 'rev_wav_et.txt']
td_name = ['REVERB_dt_ITD_mat.mat', 'REVERB_et_ITD_mat.mat']
nSample = [742, 1088]

for i in range(len(r_list)):
    print(r_list[i] + ' --> ' + w1_list[i] + ' && ' + w2_list[i])
    fp_r = open(r_list[i], 'r')
    fp_w1 = open(w1_list[i], 'w')
    fp_w2 = open(w2_list[i], 'w')

    fp_scp = open(scp_list[i], 'r')

    # step 1) make dictionary (key: wav path, value: time delay)
    gt_dict = {}
    est_dict = {}
    td_table = sio.loadmat(td_name[i])
    gt_table = td_table['gt']
    est_table = td_table['est']
    scp_lines = fp_scp.readlines()

    print('convert time delay table to dictionary')
    for n in range(2):
        if(n == 0):
            dist_prefix = 'near_test'
        elif(n == 1):
            dist_prefix = 'far_test'

        for m in trange(0, nSample[i]):
            l = nSample[i]*(n-1) + m

            path_key = dist_prefix + scp_lines[l][:-1] # exclude newline character

            gt_dict[(path_key, 2)] = gt_table[l][0]
            gt_dict[(path_key, 3)] = gt_table[l][1]
            gt_dict[(path_key, 4)] = gt_table[l][2]
            gt_dict[(path_key, 5)] = gt_table[l][3]
            gt_dict[(path_key, 6)] = gt_table[l][4]
            gt_dict[(path_key, 7)] = gt_table[l][5]
            gt_dict[(path_key, 8)] = gt_table[l][6]

            est_dict[(path_key, 2)] = est_table[l][0]
            est_dict[(path_key, 3)] = est_table[l][1]
            est_dict[(path_key, 4)] = est_table[l][2]
            est_dict[(path_key, 5)] = est_table[l][3]
            est_dict[(path_key, 6)] = est_table[l][4]
            est_dict[(path_key, 7)] = est_table[l][5]
            est_dict[(path_key, 8)] = est_table[l][6]


    print('append to the manifest')
    lines = fp_r.readlines()

    #print('path_key = ' + path_key)
    #print(' ')
    for k in trange(0, len(lines)):
    #for line in lines:
        line = lines[k]
        line_splited = line.split(',')
        noisy_path = line_splited[0]
        clean_path = line_splited[1][:-1] # exclude newline character
        noisy_path_splited = noisy_path.split('/')
        path_key_find = '/'.join(noisy_path_splited[-5:])


        #print('path_key_find = ' + path_key_find)

        # append ground truth time delay
        #pdb.set_trace()
        gt_td12 = str(gt_dict[(path_key_find, 2)])
        gt_td13 = str(gt_dict[(path_key_find, 3)])
        gt_td14 = str(gt_dict[(path_key_find, 4)])
        gt_td15 = str(gt_dict[(path_key_find, 5)])
        gt_td16 = str(gt_dict[(path_key_find, 6)])
        gt_td17 = str(gt_dict[(path_key_find, 7)])
        gt_td18 = str(gt_dict[(path_key_find, 8)])

        line_w1 = noisy_path + ',' + clean_path + ',' + gt_td12 + ',' + gt_td13 + ',' + gt_td14 + ',' + gt_td15 + ',' + gt_td16 + ',' + gt_td17 + ',' + gt_td18 + '\n'
        fp_w1.write(line_w1)

        # append estimated time delay
        est_td12 = str(est_dict[(path_key_find, 2)])
        est_td13 = str(est_dict[(path_key_find, 3)])
        est_td14 = str(est_dict[(path_key_find, 4)])
        est_td15 = str(est_dict[(path_key_find, 5)])
        est_td16 = str(est_dict[(path_key_find, 6)])
        est_td17 = str(est_dict[(path_key_find, 7)])
        est_td18 = str(est_dict[(path_key_find, 8)])

        line_w2 = noisy_path + ',' + clean_path + ',' + est_td12 + ',' + est_td13 + ',' + est_td14 + ',' + est_td15 + ',' + est_td16 + ',' + est_td17 + ',' + est_td18 + '\n'
        fp_w2.write(line_w2)

    print(' ')
    print(' ')

    fp_scp.close()
    fp_r.close()
    fp_w1.close()
    fp_w2.close()

'''

# Train
'''
r_list = ['reverb_tr_simu_8ch_paired.csv']
w1_list = ['reverb_tr_simu_8ch_td_gt.csv']
w2_list = ['reverb_tr_simu_8ch_td_est.csv']
scp_list = ['rev_wav_tr.txt']
td_name = ['REVERB_tr_ITD_mat.mat']
nSample = [7861]

for i in range(len(r_list)):
    print(r_list[i] + ' --> ' + w1_list[i] + ' && ' + w2_list[i])
    fp_r = open(r_list[i], 'r')
    fp_w1 = open(w1_list[i], 'w')
    fp_w2 = open(w2_list[i], 'w')

    fp_scp = open(scp_list[i], 'r')

    # step 1) make dictionary (key: wav path, value: time delay)
    gt_dict = {}
    est_dict = {}
    td_table = sio.loadmat(td_name[i])
    gt_table = td_table['gt']
    est_table = td_table['est']
    scp_lines = fp_scp.readlines()

    print('convert time delay table to dictionary')
    for l in trange(0, nSample[i]):
        path_key = scp_lines[l][1:-1] # exclude newline character & first /

        gt_dict[(path_key, 2)] = gt_table[l][0]
        gt_dict[(path_key, 3)] = gt_table[l][1]
        gt_dict[(path_key, 4)] = gt_table[l][2]
        gt_dict[(path_key, 5)] = gt_table[l][3]
        gt_dict[(path_key, 6)] = gt_table[l][4]
        gt_dict[(path_key, 7)] = gt_table[l][5]
        gt_dict[(path_key, 8)] = gt_table[l][6]

        est_dict[(path_key, 2)] = est_table[l][0]
        est_dict[(path_key, 3)] = est_table[l][1]
        est_dict[(path_key, 4)] = est_table[l][2]
        est_dict[(path_key, 5)] = est_table[l][3]
        est_dict[(path_key, 6)] = est_table[l][4]
        est_dict[(path_key, 7)] = est_table[l][5]
        est_dict[(path_key, 8)] = est_table[l][6]


    print('append to the manifest')
    lines = fp_r.readlines()

    #print('path_key = ' + path_key)
    #print(' ')
    for k in trange(0, len(lines)):
    #for line in lines:
        line = lines[k]
        line_splited = line.split(',')
        noisy_path = line_splited[0]
        clean_path = line_splited[1][:-1] # exclude newline character
        noisy_path_splited = noisy_path.split('/')
        path_key_find = '/'.join(noisy_path_splited[-4:])
        #print('path_key_find = ' + path_key_find)

    # append ground truth time delay
        #pdb.set_trace()
        gt_td12 = str(gt_dict[(path_key_find, 2)])
        gt_td13 = str(gt_dict[(path_key_find, 3)])
        gt_td14 = str(gt_dict[(path_key_find, 4)])
        gt_td15 = str(gt_dict[(path_key_find, 5)])
        gt_td16 = str(gt_dict[(path_key_find, 6)])
        gt_td17 = str(gt_dict[(path_key_find, 7)])
        gt_td18 = str(gt_dict[(path_key_find, 8)])

        line_w1 = noisy_path + ',' + clean_path + ',' + gt_td12 + ',' + gt_td13 + ',' + gt_td14 + ',' + gt_td15 + ',' + gt_td16 + ',' + gt_td17 + ',' + gt_td18 + '\n'
        fp_w1.write(line_w1)

        # append estimated time delay
        est_td12 = str(est_dict[(path_key_find, 2)])
        est_td13 = str(est_dict[(path_key_find, 3)])
        est_td14 = str(est_dict[(path_key_find, 4)])
        est_td15 = str(est_dict[(path_key_find, 5)])
        est_td16 = str(est_dict[(path_key_find, 6)])
        est_td17 = str(est_dict[(path_key_find, 7)])
        est_td18 = str(est_dict[(path_key_find, 8)])

        line_w2 = noisy_path + ',' + clean_path + ',' + est_td12 + ',' + est_td13 + ',' + est_td14 + ',' + est_td15 + ',' + est_td16 + ',' + est_td17 + ',' + est_td18 + '\n'
        fp_w2.write(line_w2)


    fp_scp.close()
    fp_r.close()
    fp_w1.close()
    fp_w2.close()
'''

# dt & et - trainIR

r_list = ['reverb_dt_simu_8ch_trainIR_paired.csv', 'reverb_et_hard6cases_simu_8ch_trainIR_paired.csv']
w1_list = ['reverb_dt_simu_8ch_trainIR_td_gt.csv', 'reverb_et_hard6cases_simu_8ch_trainIR_td_gt.csv']
w2_list = ['reverb_dt_simu_8ch_trainIR_td_est.csv', 'reverb_et_hard6cases_simu_8ch_trainIR_td_est.csv']
scp_list = ['rev_wav_dt.txt', 'rev_wav_et.txt']
td_name = ['REVERB_dt_ITD_mat.mat', 'REVERB_et_ITD_mat.mat']
nSample = [742, 1088]

replace_list_src = ['far_test/si_dt', 'far_test/si_et_1', 'far_test/si_et_2',
                    'near_test/si_dt', 'near_test/si_et_1', 'near_test/si_et_2']
replace_list_tgt = ['far_test/primary_microphone/si_dt', 'far_test/primary_microphone/si_et_1', 'far_test/secondary_microphone/si_et_2',
                    'near_test/primary_microphone/si_dt', 'near_test/primary_microphone/si_et_1', 'near_test/secondary_microphone/si_et_2',]


for i in range(len(r_list)):
    print(r_list[i] + ' --> ' + w1_list[i] + ' && ' + w2_list[i])
    fp_r = open(r_list[i], 'r')
    fp_w1 = open(w1_list[i], 'w')
    fp_w2 = open(w2_list[i], 'w')

    fp_scp = open(scp_list[i], 'r')

    # step 1) make dictionary (key: wav path, value: time delay)
    gt_dict = {}
    est_dict = {}
    td_table = sio.loadmat(td_name[i])
    gt_table = td_table['gt']
    est_table = td_table['est']
    scp_lines = fp_scp.readlines()

    print('convert time delay table to dictionary')
    for n in range(2):
        if(n == 0):
            dist_prefix = 'near_test'
        elif(n == 1):
            dist_prefix = 'far_test'

        for m in trange(0, nSample[i]):
            l = nSample[i]*(n-1) + m

            path_key = dist_prefix + scp_lines[l][:-1] # exclude newline character

            gt_dict[(path_key, 2)] = gt_table[l][0]
            gt_dict[(path_key, 3)] = gt_table[l][1]
            gt_dict[(path_key, 4)] = gt_table[l][2]
            gt_dict[(path_key, 5)] = gt_table[l][3]
            gt_dict[(path_key, 6)] = gt_table[l][4]
            gt_dict[(path_key, 7)] = gt_table[l][5]
            gt_dict[(path_key, 8)] = gt_table[l][6]

            est_dict[(path_key, 2)] = est_table[l][0]
            est_dict[(path_key, 3)] = est_table[l][1]
            est_dict[(path_key, 4)] = est_table[l][2]
            est_dict[(path_key, 5)] = est_table[l][3]
            est_dict[(path_key, 6)] = est_table[l][4]
            est_dict[(path_key, 7)] = est_table[l][5]
            est_dict[(path_key, 8)] = est_table[l][6]


    print('append to the manifest')
    lines = fp_r.readlines()

    #print('path_key = ' + path_key)
    #print(' ')
    for k in trange(0, len(lines)):
    #for line in lines:
        line = lines[k]
        line_splited = line.split(',')
        noisy_path = line_splited[0]
        clean_path = line_splited[1][:-1] # exclude newline character
        noisy_path_splited = noisy_path.split('/')
        path_key_find_prev = '/'.join(noisy_path_splited[-3:])

        for h in range(len(replace_list_src)):
            path_key_find_prev = path_key_find_prev.replace(replace_list_src[h], replace_list_tgt[h])

        path_key_find_prev_splited = path_key_find_prev.split('/')
        path_key_find = path_key_find_prev_splited[0] + '/' + path_key_find_prev_splited[1] + '/' + path_key_find_prev_splited[2] + '/' + path_key_find_prev_splited[3][:3] + '/' + path_key_find_prev_splited[3]

        #print('path_key_find = ' + path_key_find)

        # append ground truth time delay
        #pdb.set_trace()
        gt_td12 = str(gt_dict[(path_key_find, 2)])
        gt_td13 = str(gt_dict[(path_key_find, 3)])
        gt_td14 = str(gt_dict[(path_key_find, 4)])
        gt_td15 = str(gt_dict[(path_key_find, 5)])
        gt_td16 = str(gt_dict[(path_key_find, 6)])
        gt_td17 = str(gt_dict[(path_key_find, 7)])
        gt_td18 = str(gt_dict[(path_key_find, 8)])

        line_w1 = noisy_path + ',' + clean_path + ',' + gt_td12 + ',' + gt_td13 + ',' + gt_td14 + ',' + gt_td15 + ',' + gt_td16 + ',' + gt_td17 + ',' + gt_td18 + '\n'
        fp_w1.write(line_w1)

        # append estimated time delay
        est_td12 = str(est_dict[(path_key_find, 2)])
        est_td13 = str(est_dict[(path_key_find, 3)])
        est_td14 = str(est_dict[(path_key_find, 4)])
        est_td15 = str(est_dict[(path_key_find, 5)])
        est_td16 = str(est_dict[(path_key_find, 6)])
        est_td17 = str(est_dict[(path_key_find, 7)])
        est_td18 = str(est_dict[(path_key_find, 8)])

        line_w2 = noisy_path + ',' + clean_path + ',' + est_td12 + ',' + est_td13 + ',' + est_td14 + ',' + est_td15 + ',' + est_td16 + ',' + est_td17 + ',' + est_td18 + '\n'
        fp_w2.write(line_w2)

    print(' ')
    print(' ')

    fp_scp.close()
    fp_r.close()
    fp_w1.close()
    fp_w2.close()