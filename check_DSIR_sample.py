from tqdm import trange
#import soundfile as sf
import os

nCH = 8
manifest_path = 'data_sorted/L221_10105_grid10cm_tr_exists_only.csv'
manifest = open(manifest_path, 'r')
CH_non_equal_sample_list = open('CH_non_equal_sample_list.txt', 'w')
lines = manifest.readlines()

for i in trange(0, len(lines)):
    line = lines[i]
    reverb_prefix = line.split(',')[0]
    reverb_path_ch1 = reverb_prefix + '_ch' + str(1) + '.wav'
    # ver 1
    #reverb_ch1, sr = sf.read(reverb_path_ch1)
    #len_ch1 = reverb_ch1.shape[0]

    # ver 2
    sz_ch1 = os.path.getsize(reverb_path_ch1)
    for c in range(1, nCH):
        reverb_path = reverb_prefix + '_ch' + str(c+1) + '.wav'
        # ver 1
        #reverb, sr = sf.read(reverb_path)
        #len_ch = reverb.shape[0]
        #if(len_ch != len_ch1):

        # ver2
        sz_ch = os.path.getsize(reverb_path)
        if(sz_ch != sz_ch1):
            print('found one, save name!')
            print(reverb_prefix)
            CH_non_equal_sample_list.write(reverb_prefix + '\n')
            break

manifest.close()
CH_non_equal_sample_list.close()