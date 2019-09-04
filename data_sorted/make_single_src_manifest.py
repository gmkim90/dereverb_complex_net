from tqdm import trange

manifest_r_list = ['L553_fixedmic_hyper_0.1cm_RT0.2_nongrid_0.1.csv', 'L553_fixedmic_hyper_0.1cm_RT0.2_nongrid_0.5.csv']
manifest_w_list = ['L553_fixedmic_hyper_0.1cm_nongrid_fixed5src.csv', 'L553_fixedmic_hyper_0.5cm_nongrid_fixed5src.csv']

fixed_src_path = '/home/kenkim/28-12332-0025+4926-23311-0001+ 6963-81511-0081+ 986-129388-0094+1001-134707-0000.wav'

for i in range(len(manifest_r_list)):
    manifest_r = open(manifest_r_list[i], 'r')
    manifest_w = open(manifest_w_list[i], 'w')
    print(manifest_r_list[i] + ' --> ' + manifest_w_list[i])

    lines = manifest_r.readlines()
    for j in trange(0, len(lines)):
        line = lines[j]
        line_splited = line.split(',')
        line_w = line_splited[0] + ',' + fixed_src_path
        manifest_w.write(line_w + '\n')

    manifest_r.close()
    manifest_w.close()