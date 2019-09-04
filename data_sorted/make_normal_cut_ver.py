r_list = ['reverb_dt_simu_8ch_paired_BACKUP.csv', 'reverb_et_simu_8ch_paired_BACKUP.csv', 'reverb_et_hard6cases_simu_8ch_paired_BACKUP.csv']
w1_list = ['reverb_dt_simu_8ch_cut_paired.csv', 'reverb_et_simu_8ch_cut_paired.csv', 'reverb_et_hard6cases_simu_8ch_cut_paired.csv']
w2_list = ['reverb_dt_simu_8ch_paired.csv', 'reverb_et_simu_8ch_paired.csv', 'reverb_et_hard6cases_simu_8ch_paired.csv']

for i in range(len(r_list)):
    print(r_list[i] + ' --> ' + w1_list[i])
    print(r_list[i] + ' --> ' + w2_list[i])
    print('')
    fp_r = open(r_list[i], 'r')
    fp_w1 = open(w1_list[i], 'w')
    fp_w2 = open(w2_list[i], 'w')

    lines = fp_r.readlines()

    for line in lines:
        line_splited = line.split(',')
        noisy_path = line_splited[0]
        clean_path = line_splited[1]

        # cut version (w1 list)
        noisy_path_splited = noisy_path.split('/')
        noisy_path_splited[-2] = ''
        noisy_path_cut = '/'.join(noisy_path_splited)
        noisy_path_cut = noisy_path_cut.replace('//', '/')
        line_w1 = noisy_path_cut + ',' + clean_path
        fp_w1.write(line_w1)

        # uncut version (w2 list)
        noisy_path = noisy_path.replace('_cut', '')
        noisy_path = noisy_path.replace('si_dt', 'primary_microphone/si_dt')
        noisy_path = noisy_path.replace('si_et_1', 'primary_microphone/si_et_1')
        noisy_path = noisy_path.replace('si_et_2', 'primary_microphone/si_et_2')
        noisy_path_splited = noisy_path.split('/')
        #noisy_path_splited[-1] = noisy_path_splited[-1][:3] + '/' + noisy_path_splited[-1]
        noisy_path_new = '/'.join(noisy_path_splited)

        line_w2 = noisy_path_new + ',' + clean_path
        fp_w2.write(line_w2)

    fp_r.close()
    fp_w1.close()
    fp_w2.close()