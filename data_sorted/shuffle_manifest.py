from shutil import copyfile
import os
import random

prefix = 'L553_30cm_30cm_1cm_nSrc_'

r_list = [prefix + '10_TMP', prefix + '50_TMP', prefix + '100_TMP', prefix + '150_TMP', prefix + '200_TMP',
          prefix + '100_ref1_oto_TMP', prefix + '100_ref1_ofa_TMP', prefix + '100_ref2_oto_TMP', prefix + '100_ref2_ofa_TMP']
w_list = [prefix + '10', prefix + '50', prefix + '100', prefix + '150', prefix + '200',
          prefix + '100_ref1_oto', prefix + '100_ref1_ofa', prefix + '100_ref2_oto', prefix + '100_ref2_ofa']

# Make temporary file
for i in range(len(r_list)):
    if(not os.path.exists(r_list[i])):
        copyfile(w_list[i]+'.csv', r_list[i]+'.csv')

# Shuffle manifest & write
for i in range(len(r_list)):
    print(r_list[i] + ' --> ' + w_list[i])
    manifest_r = open(r_list[i] + '.csv', 'r')
    manifest_w = open(w_list[i] + '.csv', 'w')

    lines_r = manifest_r.readlines()
    random.shuffle(lines_r)

    for j in range(len(lines_r)):
        manifest_w.write(lines_r[j].strip() + '\n')

    manifest_r.close()
    manifest_w.close()