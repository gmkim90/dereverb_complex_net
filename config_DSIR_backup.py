import argparse
import os

import pdb

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser()
parser.add_argument('--model_json', default='experiments/base_model', help="Directory containing params.json")
#parser.add_argument('--gpu', default=-1, type=int, help="gpuid (>=0)") # use CUDA_VISIBLE_DEVICES is more safe way
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_json containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--tr_manifest', default='data_sorted/reverb_tr_simu_8ch_paired.csv', type=str)
parser.add_argument('--trsub_manifest', default='data_sorted/reverb_trsub_simu_8ch_paired.csv', type=str)
parser.add_argument('--val_manifest', default='data_sorted/reverb_dt_simu_8ch_paired.csv', type=str)
parser.add_argument('--te_smallset_manifest', default='data_sorted/reverb_et_hard6cases_simu_8ch_paired.csv', type=str)

parser.add_argument('--val_cut_manifest', default='', type=str)
parser.add_argument('--te_smallset_cut_manifest', default='', type=str)

#parser.add_argument('--val_trainIR_manifest', default='data_sorted/reverb_dt_simu_8ch_trainIR_paired.csv', type=str)
#parser.add_argument('--te_smallset_trainIR_manifest', default='data_sorted/reverb_et_hard6cases_simu_8ch_trainIR_paired.csv', type=str)
parser.add_argument('--val_trainIR_manifest', default='', type=str)
parser.add_argument('--te_smallset_trainIR_manifest', default='', type=str)


parser.add_argument('--nMic', default=8, type=int, help='#microphone')
parser.add_argument('--reverb_frame', default=1, type=int, help='set reverb_frame > 1 for multi-frame demixing weight, reverb_frame = 1 for single frame')
parser.add_argument('--mic_sampling', default='no', type=str, help='no | random | ref_random | ref_manual')
parser.add_argument('--subset1', default=None, type=str, help='e.g) "1,2,3"')
parser.add_argument('--subset2', default=None, type=str, help='e.g) "1,2,3"')
parser.add_argument('--do_eval', default=True, type=str2bool)
parser.add_argument('--nFFT', default=1024, type=int)
parser.add_argument('--nWin', default=1024, type=int)
parser.add_argument('--use_BN', default=True, type=str2bool)
parser.add_argument('--input_type', default='complex', type=str, help='complex (mic)| complex_ratio (mic ratio) | log_complex_ratio | IPD_preprocess')
parser.add_argument('--td_type', default='', type=str, help='gt|est|blank')
parser.add_argument('--measure_reverb_loss', default=False, type=str2bool)
parser.add_argument('--ds_rate', default=1, type=int, help = 'integer with power of 2')
parser.add_argument('--vad_preemp', default=False, type=str2bool, help = 'apply vad+preemp to mic, vad to clean')
parser.add_argument('--fix_len_by_cl', default='eval', type=str, help = 'input | eval')
parser.add_argument('--fband_SDR', default=False, type=str2bool)
parser.add_argument('--nFinterval', default=9, type=int) # 10 ->9
parser.add_argument('--use_depthwise', default=False, type=str2bool)
parser.add_argument('--w_var', default=0, type=float)

# DSIR dataset
parser.add_argument('--grid_cm', default=0, type=int, help = 'if larger than 0, it use L221_10105_grid dataset ')
parser.add_argument('--nSource', default=2, type=int, help = 'nSource per IR (default=2), choose 1 or 2')
parser.add_argument('--augment', default='no', type=str, help = 'no(default)|IR|Src')


parser.add_argument('--model_type', default = 'unet', type=str, help = 'unet|lcn')
parser.add_argument('--nHidden', default=64, type=int) # LCN
parser.add_argument('--nLayer', default=3, type=int) # LCN
parser.add_argument('--ksz_time', default=3, type=int) # LCN
parser.add_argument('--CW_freq', default=0, type=int) # LCN

parser.add_argument('--fs', default=16000, type=int)
parser.add_argument('--f_start', default=0, type=int)
parser.add_argument('--f_end', default=8000, type=int) # nyquist freq
parser.add_argument('--save_tr_wav', default=False, type=str2bool)


#parser.add_argument('--power_reciprocal_conjugate', default=False, type=str2bool, help='W = power_reciprocal_conjugate(H) as fixed solution for uniqueness')
parser.add_argument('--inverse_type', default='no', type=str, help = 'left | powre_reciprocal_conjugate | no (default)')

parser.add_argument('--mode', default='train', type=str, help = 'train | test | generate')
parser.add_argument('--nGenerate', default=1, type=int, help = 'how many minibatch to generate for each tr/dt/et')
parser.add_argument('--use_ISTFT', default=False, type=str2bool)




# for evaluation only
parser.add_argument('--eval_type', default='sInvSDR_mag', type=str, help='time|spec|from_loss')
parser.add_argument('--eval_train', default=False, type=str2bool)
parser.add_argument('--eval_val', default=False, type=str2bool)
parser.add_argument('--eval_test', default=False, type=str2bool)

parser.add_argument('--cut_dtet', default=False, type=str2bool)
parser.add_argument('--loss_type', default='sInvSDR_mag', type=str, help='cossim_time | cossim_spec | cossim_mag | SDR_time')
parser.add_argument('--batch_size', default=8, type=int, help='train batch size')
parser.add_argument('--expnum', default=-1, type=int)
parser.add_argument('--num_epochs', default=500, type=int, help='train epochs number')
parser.add_argument('--log_iter', default=10, type=int)
parser.add_argument('--eval_iter', default=500, type=int)
parser.add_argument('--lR0', default=1e-4, type=float)




args = parser.parse_args()

def get_config():
    config, unparsed = parser.parse_known_args()
    if(len(unparsed) > 0):
        print(unparsed)
        assert(len(unparsed) == 0), 'length of unparsed option should be 0'
    return config, unparsed

