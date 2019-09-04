from tqdm import trange
import os

manifest_r_name_list = ['L553_fixedmic_omni_1cm_RT0.2_tr.csv', 'L553_fixedmic_hyper_1cm_RT0.2_tr.csv']
manifest_w_name_list = ['L553_fixedmic_omni_1cm_RT0.2_trsub_zfix_xyreduce', 'L553_fixedmic_hyper_1cm_RT0.2_trsub_zfix_xyreduce']

range_xyz = [[4.0, 4.3], [4.0, 4.3], [1.3, 1.3]]

for i in range(len(manifest_r_name_list)):
    manifest_r_name = manifest_r_name_list[i]
    manifest_w_name = manifest_w_name_list[i]

    print(manifest_r_name + ' --> ' + manifest_w_name)

    manifest_r = open(manifest_r_name, 'r')
    manifest_w_src1 = open(manifest_w_name + '_src1.csv', 'w')
    manifest_w_src2 = open(manifest_w_name + '_src2.csv', 'w')

    lines = manifest_r.readlines()
    src1_list = []
    src2_list = []
    unique_IR_list = []

    print('filtering list')
    for j in trange(0, len(lines)):
        line = lines[j]
        line_splited = line.split(',')
        IR_path = line_splited[0]
        sIdx = IR_path.find('s=')
        RTIdx = IR_path.find('RT0')

        src_pos_str = IR_path[sIdx+2:RTIdx-1]
        src_pos_str_list = src_pos_str.split('_')
        src_pos = [float(f) for f in src_pos_str_list]

        # if valid position
        if(src_pos[0] >= range_xyz[0][0] and src_pos[0] <= range_xyz[0][1]
        and src_pos[1] >= range_xyz[1][0] and src_pos[1] <= range_xyz[1][1]
        and src_pos[2] >= range_xyz[2][0] and src_pos[2] <= range_xyz[2][1]):
            if(src_pos_str in unique_IR_list):
                src2_list.append(line)
            else:
                unique_IR_list.append(src_pos_str)
                src1_list.append(line)

    print('#src1 = ' + str(len(src1_list)))
    print('#src2 = ' + str(len(src2_list)))

    assert(len(src1_list) == len(src2_list))

    print('write manifest')
    for j in trange(0, len(src1_list)):
        manifest_w_src1.write(src1_list[j])
        manifest_w_src2.write(src2_list[j])

    manifest_r.close()
    manifest_w_src1.close()
    manifest_w_src2.close()