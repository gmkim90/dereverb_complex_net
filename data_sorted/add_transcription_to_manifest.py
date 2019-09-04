from tqdm import trange
import os

# add transcription to manifest

#manifest_r_list = ['reverb_tr_simu_8ch_paired.csv', 'reverb_dt_simu_8ch_paired.csv', 'diverse_room_tr.csv', 'diverse_room_val.csv']
#manifest_w_list = ['reverb_tr_simu_8ch_txt.csv', 'reverb_dt_simu_8ch_txt.csv', 'diverse_room_tr_txt.csv', 'diverse_room_val_txt.csv']
manifest_r_list = ['diverse_room_tr.csv', 'diverse_room_val.csv', 'diverse_room_te.csv']
manifest_w_list = ['diverse_room_tr_txt.csv', 'diverse_room_val_txt.csv', 'diverse_room_te_txt.csv']


# source of transcription
# REVERB : wsj0
# diverse room : librispeech
source_dir_prefix = '/home/kenkim/librispeech/txt'
source_dir_list = [source_dir_prefix + '/train', source_dir_prefix + '/val', source_dir_prefix + '/test_clean']

for n in range(len(manifest_r_list)):
    source_dir = source_dir_list[n]
    manifest_r = open(manifest_r_list[n], 'r')
    manifest_w = open(manifest_w_list[n], 'w')

    print(manifest_r_list[n] + ' --> ' + manifest_w_list[n])

    lines = manifest_r.readlines()

    for i in trange(0, len(lines)):
        line = lines[i]
        line_splited = line.split(',')
        reverb_path, clean_path = line_splited[0], line_splited[1]
        clean_path = clean_path.replace('\n', '')
        clean_id = clean_path.split('/')[-1].split('.')[0]
        txt_path = source_dir + '/' + clean_id + '.txt'
        if(not os.path.exists(reverb_path + '_ch8.wav')):
            print(reverb_path + ' not exists')
        if (not os.path.exists(clean_path)):
            print(clean_path + ' not exists')
        if (not os.path.exists(txt_path)):
            print(txt_path + ' not exists')

        line_w = reverb_path + ',' + clean_path + ',' + txt_path
        manifest_w.write(line_w + '\n')

    manifest_r.close()
    manifest_w.close()
