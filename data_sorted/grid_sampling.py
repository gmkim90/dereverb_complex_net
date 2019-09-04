from tqdm import trange
import os

# remove non-existstent files from manifest

manifest_r_list = ['L221_10105_grid10cm_val_exists_only.csv', 'L221_10105_grid10cm_te_exists_only.csv', 'L221_10105_grid10cm_tr_exists_only.csv', 'L221_10105_grid10cm_tr_exists_only.csv', 'L221_10105_grid10cm_tr_exists_only.csv']
manifest_w_list = ['L221_10105_grid10cm_val_1per.csv', 'L221_10105_grid10cm_te_1per.csv', 'L221_10105_grid20cm_tr_exists_only.csv', 'L221_10105_grid50cm_tr_exists_only.csv', 'L221_10105_grid80cm_tr_exists_only.csv']
sampling_interval_list = [100, 100, 8, 125, 512]

for n in range(len(manifest_r_list)):
    manifest_r = open(manifest_r_list[n], 'r')
    manifest_w = open(manifest_w_list[n], 'w')
    sampling_interval = sampling_interval_list[n]

    print(manifest_r_list[n] + ' --> ' + manifest_w_list[n])

    lines = manifest_r.readlines()

    for i in trange(0, len(lines)):
        line = lines[i]
        residual = i % sampling_interval
        if(residual == 0 or residual == 1):
            line_w = line.replace('\n', '')
            manifest_w.write(line_w + '\n')

    manifest_r.close()
    manifest_w.close()
