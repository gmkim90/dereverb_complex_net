from tqdm import trange

manifest_r_list = ['L553_fixedmic_hyper_1cm_RT0.2_tr.csv', 'L553_fixedmic_hyper_1cm_RT0.2_val.csv',
                   'L553_fixedmic_hyper_0.1cm_RT0.2_nongrid_0.1.csv', 'L553_fixedmic_hyper_0.1cm_RT0.2_nongrid_0.5.csv']
manifest_w_list = ['L553_fixedmic_hyper_1cm_tr_fixedsrc.csv', 'L553_fixedmic_hyper_1cm_val_fixedsrc.csv',
                   'L553_fixedmic_hyper_0.1cm_nongrid_fixedsrc.csv', 'L553_fixedmic_hyper_0.5cm_nongrid_fixedsrc.csv']

fixed_src1_path = '/home/kenkim/28-12332-0025+4926-23311-0001+ 6963-81511-0081+ 986-129388-0094+1001-134707-0000.wav'
fixed_src2_path = '/home/kenkim/4267-287369-0000+5519-39478-0000+7769-99397-0001+978-125137-0033+7789-103120-0003.wav'


for i in range(len(manifest_r_list)):
    manifest_r = open(manifest_r_list[i], 'r')
    manifest_w = open(manifest_w_list[i], 'w')
    print(manifest_r_list[i] + ' --> ' + manifest_w_list[i])

    lines = manifest_r.readlines()
    IR_unique_list = []
    for j in trange(0, len(lines)):
        line = lines[j]
        line_splited = line.split(',')
        IR_path = line_splited[0]
        if(IR_path in IR_unique_list):
            line_w = IR_path + ',' + fixed_src2_path
        else:
            IR_unique_list.append(IR_path)
            line_w = IR_path + ',' + fixed_src1_path
        manifest_w.write(line_w + '\n')

    manifest_r.close()
    manifest_w.close()