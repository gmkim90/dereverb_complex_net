from tqdm import trange
manifest_r_list = ['L221_10105_grid10cm_tr.csv', 'L221_10105_grid10cm_val.csv', 'L221_10105_grid10cm_te.csv']
manifest_w_list = ['L221_10105_grid10cm_1source_tr.csv', 'L221_10105_grid10cm_1source_val.csv', 'L221_10105_grid10cm_1source_te.csv']

for i in range(len(manifest_r_list)):
	manifest_r = open(manifest_r_list[i], 'r')
	manifest_w = open(manifest_w_list[i], 'w')

	print('writing ' + manifest_w_list[i])

	ID_list = []
	lines = manifest_r.readlines()
	for j in trange(0, len(lines)):
		line = lines[j]
		k1 = line.find('=')
		k2 = line.find('_RT')
		ID = line[k1+1:k2]

		if(ID not in ID_list):
			ID_list.append(ID)
			manifest_w.write(line)

	manifest_w.close()
	manifest_r.close()
