from tqdm import trange

r_list = ['L221_10105_grid10cm_tr.csv', 'L221_10105_grid10cm_val.csv']
nSample_IR = 4

for r in range(len(r_list)):
    print('searching ' + r_list[r])
    manifest = open(r_list[r], 'r')
    lines = manifest.readlines()

    posIdx_nSample_2_list = [] # final
    posIdx_all_list = []
    for i in trange(0, len(lines)):
        line = lines[i]
        k1 = line.find('=')
        k2 = line.find('_RT')
        posIdx = line[k1+1:k2]

        if(posIdx not in posIdx_all_list):
            posIdx_all_list.append(posIdx)
        else:
            posIdx_all_list.append(posIdx)
            posIdx_nSample_2_list.append(posIdx)

        if(len(posIdx_nSample_2_list) == nSample_IR):
            print('found ' + str(nSample_IR) + ' posIdx with 2 samples')
            print(posIdx_nSample_2_list)
            break

    manifest.close()