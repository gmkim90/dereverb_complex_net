from tqdm import trange

manifest_train_list = ['L553_30cm_30cm_1cm_nSrc_100.csv', 'L553_30cm_30cm_1cm_nSrc_100.csv', 'L553_30cm_30cm_1cm_nSrc_100.csv', 'L553_30cm_30cm_1cm_nSrc_100.csv']
manifest_reference_list = ['L553_pos961_ref1_IRonly_RT0.2.csv', 'L553_pos961_ref1_IRonly_RT0.2.csv', 'L553_pos961_ref2_IRonly_RT0.2.csv', 'L553_pos961_ref2_IRonly_RT0.2.csv']
ref_type_list = ['one-to-one', 'one-for-all', 'one-to-one', 'one-for-all']
manifest_train_ref_list = ['L553_30cm_30cm_1cm_nSrc_100_ref1_oto.csv', 'L553_30cm_30cm_1cm_nSrc_100_ref1_ofa.csv', 'L553_30cm_30cm_1cm_nSrc_100_ref2_oto.csv', 'L553_30cm_30cm_1cm_nSrc_100_ref2_ofa.csv']
#	oto: one-to-one (1target = 1reference  #Reference = 961)
#	ofa: one-for-all (961target = 1reference  #Refernece = 1)

nSrc = 100

for m in range(len(manifest_train_ref_list)):
    print('combine ' + manifest_train_list[m] + ' and '  + manifest_reference_list[m])
    print('to make ' + manifest_train_ref_list[m])
    print('')

    manifest_train = open(manifest_train_list[m], 'r')
    manifest_reference = open(manifest_reference_list[m], 'r')
    manifest_train_ref = open(manifest_train_ref_list[m], 'w')

    ref_type = ref_type_list[m]

    manifest_train_lines = manifest_train.readlines()
    manifest_reference_lines = manifest_reference.readlines()

    for n in trange(0, len(manifest_train_lines)):
        tr_line = manifest_train_lines[n]
        if(ref_type == 'one-for-all'): # w.o.l.g, choose 1st IR in reference position
            ref_line = manifest_reference_lines[0]
        elif(ref_type == 'one-to-one'):
            ref_idx = n//nSrc # be aware of using '//' in python3.7
            ref_line = manifest_reference_lines[ref_idx]
        line_w = tr_line.strip() + ',' + ref_line.strip()
        manifest_train_ref.write(line_w + '\n')

    manifest_train.close()
    manifest_reference.close()
    manifest_train_ref.close()
