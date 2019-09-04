from tqdm import trange
import os

# remove non-existstent files from manifest

manifest_r_list = ['L221_10105_grid10cm_tr.csv', 'L221_10105_grid10cm_val.csv', 'L221_10105_grid10cm_te.csv']
manifest_w_list = ['L221_10105_grid10cm_tr_exists_only.csv', 'L221_10105_grid10cm_val_exists_only.csv', 'L221_10105_grid10cm_te_exists_only.csv']

for n in range(len(manifest_r_list)):

    manifest_r = open(manifest_r_list[n], 'r')
    manifest_w = open(manifest_w_list[n], 'w')

    print(manifest_r_list[n] + ' --> ' + manifest_w_list[n])

    lines = manifest_r.readlines()

    for i in trange(0, len(lines)):
        line = lines[i]
        line_splited = line.split(',')
        reverb_path, clean_path = line_splited[0], line_splited[1]
        clean_path = clean_path.replace('\n', '')

        reverb_exists = os.path.exists(reverb_path + '_ch8.wav')
        clean_exists = os.path.exists(clean_path)

        if(reverb_exists and clean_exists): # if both file exists
            line_w = line.replace('\n', '')
            manifest_w.write(line_w + '\n')

    manifest_r.close()
    manifest_w.close()
