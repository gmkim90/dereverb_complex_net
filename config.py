import argparse
import os

import pdb

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser()
parser.add_argument('--model_json', default='experiments/base_model', help="Directory containing params.json")
#parser.add_argument('--gpu', default=-1, type=int, help="gpuid (>=0)") # # use CUDA_VISIBLE_DEVICES is more safe way
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_json containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--tr_manifest', default='', type=str)
parser.add_argument('--trsub_manifest', default='', type=str)
parser.add_argument('--val_manifest', default='', type=str)
#parser.add_argument('--te_smallset_manifest', default='data_sorted/reverb_et_hard6cases_simu_8ch_paired.csv', type=str)
parser.add_argument('--te1_manifest', default='', type=str)
parser.add_argument('--te2_manifest', default='', type=str)

parser.add_argument('--nSource', default=1, type=int)

parser.add_argument('--fix_src', default=False, type=str2bool)
parser.add_argument('--fix_src_n', default=0, type=int)
parser.add_argument('--skip_if_gen_exists', default=False, type=str2bool)


parser.add_argument('--do_log', default=True, type=str2bool)
parser.add_argument('--save_activation', default=False, type=str2bool)

parser.add_argument('--start_ratio', default=0.0, type=float)
parser.add_argument('--end_ratio', default=1.0, type=float)

parser.add_argument('--load_IR', default=True, type=str2bool)
#parser.add_argument('--idx_to_val_postfix', default='', type=str)
parser.add_argument('--pos_val_postfix', default='', type=str)
parser.add_argument('--src_range', default='0,100,0,100,0,100', type=str, help = 'string with delimiter , & splited len should be 6 or 12')
parser.add_argument('--directivity', default='hyper', type=str, help = 'omni(omnidirectional) | hyper(hypercardioid)')
parser.add_argument('--use_localization', default=False, type=str2bool)

parser.add_argument('--ec_decomposition', default=False, type=str2bool)
parser.add_argument('--carrier_input_indep', default=False, type=str2bool)
parser.add_argument('--ec_bias', default=False, type=str2bool)
parser.add_argument('--carrier_init_scale', default=0.001, type=float)



parser.add_argument('--nMic', default=8, type=int, help='#microphone')
parser.add_argument('--reverb_frame', default=1, type=int, help='set reverb_frame > 1 for multi-frame demixing weight, reverb_frame = 1 for single frame')
parser.add_argument('--mic_sampling', default='no', type=str, help='no | random | ref_random | ref_manual')
parser.add_argument('--subset1', default=None, type=str, help='e.g) "1,2,3"')
parser.add_argument('--subset2', default=None, type=str, help='e.g) "1,2,3"')
parser.add_argument('--do_eval', default=True, type=str2bool)
parser.add_argument('--nFFT', default=8192, type=int)
parser.add_argument('--nWin', default=8192, type=int)
parser.add_argument('--use_BN', default=True, type=str2bool)
parser.add_argument('--input_type', default='complex_ratio', type=str, help='complex (mic)| complex_ratio (mic ratio) | log_complex_ratio | IPD_preprocess')
parser.add_argument('--td_type', default='', type=str, help='gt|est|blank')
parser.add_argument('--measure_reverb_loss', default=False, type=str2bool)
parser.add_argument('--ds_rate', default=8, type=int, help = 'integer with power of 2')
parser.add_argument('--vad_preemp', default=False, type=str2bool, help = 'apply vad+preemp to mic, vad to clean')
parser.add_argument('--fix_len_by_cl', default='eval', type=str, help = 'input | eval')
parser.add_argument('--fband_SDR', default=False, type=str2bool)
parser.add_argument('--nFinterval', default=9, type=int) # 10 ->9
parser.add_argument('--use_depthwise', default=False, type=str2bool)
parser.add_argument('--w_var', default=0, type=float)

parser.add_argument('--grid_cm', default=0, type=int, help = 'if larger than 0, it use L221_10105_grid dataset ')

parser.add_argument('--model_type', default = 'unet', type=str, help = 'unet|lcn')
parser.add_argument('--nHidden', default=64, type=int) # LCN
parser.add_argument('--nLayer', default=3, type=int) # LCN
parser.add_argument('--ksz_time', default=3, type=int) # LCN
parser.add_argument('--CW_freq', default=0, type=int) # LCN

parser.add_argument('--fs', default=16000, type=int)
parser.add_argument('--f_start', default=0, type=int)
parser.add_argument('--f_end', default=8000, type=int) # nyquist freq
parser.add_argument('--save_wav', default=False, type=str2bool)


#parser.add_argument('--power_reciprocal_conjugate', default=False, type=str2bool, help='W = power_reciprocal_conjugate(H) as fixed solution for uniqueness')
parser.add_argument('--inverse_type', default='no', type=str, help = 'left | powre_reciprocal_conjugate | no (default)')

parser.add_argument('--mode', default='train', type=str, help = 'train | test | generate')
parser.add_argument('--nGenerate', default=1, type=int, help = 'how many minibatch to generate for each tr/dt/et')
parser.add_argument('--use_ISTFT', default=False, type=str2bool)
parser.add_argument('--return_path', default=False, type=str2bool)

parser.add_argument('--REV_VIS', default=False, type=str2bool, help = 'REVERB-Visualization scenario')
parser.add_argument('--hop_length', default=0, type=int, help = 'if 0, automatically selected as win_size/2')

parser.add_argument('--RT', default=0.2, type=float)



# for evaluation only
parser.add_argument('--eval_type', default='sInvSDR_mag', type=str, help='time|spec|sInvSDR_mag|srcIndepSDR_mag')
parser.add_argument('--eval2_type', default='', type=str, help='optional measurement')
parser.add_argument('--eval_train', default=False, type=str2bool)
parser.add_argument('--eval_val', default=False, type=str2bool)
parser.add_argument('--eval_test', default=False, type=str2bool)
parser.add_argument('--fixed_src', default=False, type=str2bool)

parser.add_argument('--cut_dtet', default=False, type=str2bool)
parser.add_argument('--loss_type', default='sInvSDR_mag', type=str, help='time|spec|sInvSDR_mag|srcIndepSDR_mag')
parser.add_argument('--loss2_type', default='', type=str, help='referece_position_demixing')

parser.add_argument('--use_ref_IR', default=False, type=str2bool, help='usage of IR from reference position')

parser.add_argument('--batch_size', default=8, type=int, help='train batch size')
parser.add_argument('--expnum', default=-1, type=int)
parser.add_argument('--num_epochs', default=500, type=int, help='train epochs number')
parser.add_argument('--log_iter', default=10, type=int)
parser.add_argument('--eval_iter', default=500, type=int, help = 'if 0, it will be set to #iter per epoch (= len(train_loader))')
parser.add_argument('--lR0', default=1e-4, type=float)
parser.add_argument('--eps', default=1e-20, type=float)
parser.add_argument('--start_epoch', default=0, type=int, help = 'if > 0, resume training from end of the train')

#parser.add_argument('--clamp_src', default=0, type=int, help = 'if > 0, clamp clean source of front/end. (i.e., clamp front/end silence)')
parser.add_argument('--clamp_frame', default=0, type=int, help = 'if > 0, clamp S & X of front/end as frame unit(i.e., clamp front/end silence)')

parser.add_argument('--ref_mic_direct_td_subtract', default=True, type=str2bool)

parser.add_argument('--interval_cm_tr', default=1, type=int)
parser.add_argument('--interval_cm_te', default=1, type=int)




args = parser.parse_args()

def get_config():
    config, unparsed = parser.parse_known_args()
    if(len(unparsed) > 0):
        print(unparsed)
        assert(len(unparsed) == 0), 'length of unparsed option should be 0'
    return config, unparsed

