r_list = ['reverb_dt_simu_8ch_paired_BACKUP.csv', 'reverb_et_simu_8ch_paired_BACKUP.csv', 'reverb_et_hard6cases_simu_8ch_paired_BACKUP.csv']
w_list = ['reverb_dt_simu_8ch_paired.csv', 'reverb_et_simu_8ch_paired.csv', 'reverb_et_hard6cases_simu_8ch_paired.csv']

for i in range(len(r_list)):
    print(r_list[i] + ' --> ' + w_list[i])
    fp_r = open(r_list[i], 'r')
    fp_w = open(w_list[i], 'w')

    lines = fp_r.readlines()

    for line in lines:
        line_splited = line.split(',')
        noisy_path = line_splited[0]
        clean_path = line_splited[1]
        noisy_path_splited = noisy_path.split('/')
        noisy_path_splited[-2] = ''
        noisy_path_new = '/'.join(noisy_path_splited)
        noisy_path_new = noisy_path_new.replace('//', '/')

        line_w = noisy_path_new + ',' + clean_path
        fp_w.write(line_w)