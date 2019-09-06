#!/bin/bash

# 10cm, IMR, batch_size = 1
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 192 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/L221_10105_grid10cm_tr_4IRs.csv --val_manifest data_sorted/L221_10105_grid10cm_val_4IRs.csv --batch_size 1 --nGenerate 8

# 10cm, Mic, batch_size = 1
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 193 --ds_rate 1 --nWin 1024 --nFFT 1024 --input_type complex --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/L221_10105_grid10cm_tr_4IRs.csv --val_manifest data_sorted/L221_10105_grid10cm_val_4IRs.csv --batch_size 1 --nGenerate 8

# 20cm, IMR, batch_size = 1
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 206 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/L221_10105_grid10cm_tr_4IRs.csv --val_manifest data_sorted/L221_10105_grid10cm_val_4IRs.csv --batch_size 1 --nGenerate 8

# 20cm, Mic, batch_size = 1
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 207 --ds_rate 1 --nWin 1024 --nFFT 1024 --input_type complex --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/L221_10105_grid10cm_tr_4IRs.csv --val_manifest data_sorted/L221_10105_grid10cm_val_4IRs.csv --batch_size 1 --nGenerate 8


# 10cm, IMR, batch_size = 8
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 192 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/L221_10105_grid10cm_tr_4IRs.csv --val_manifest data_sorted/L221_10105_grid10cm_val_4IRs.csv --batch_size 8 --nGenerate 1

# 10cm, Mic, batch_size = 8
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 193 --ds_rate 1 --nWin 1024 --nFFT 1024 --input_type complex --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/L221_10105_grid10cm_tr_4IRs.csv --val_manifest data_sorted/L221_10105_grid10cm_val_4IRs.csv --batch_size 8 --nGenerate 1

# 20cm, IMR, batch_size = 8
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 206 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/L221_10105_grid10cm_tr_4IRs.csv --val_manifest data_sorted/L221_10105_grid10cm_val_4IRs.csv --batch_size 8 --nGenerate 1

# 20cm, Mic, batch_size = 8
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 207 --ds_rate 1 --nWin 1024 --nFFT 1024 --input_type complex --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/L221_10105_grid10cm_tr_4IRs.csv --val_manifest data_sorted/L221_10105_grid10cm_val_4IRs.csv --batch_size 8 --nGenerate 1


# Visualize scenario test set, model = 10cm IMR
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 192 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/visualize_scenario_test_set.csv --val_manifest '' --batch_size 8 --nGenerate 1

# Visualize scenario test set, model = 10cm Mic
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 193 --ds_rate 1 --nWin 1024 --nFFT 1024 --input_type complex --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/visualize_scenario_test_set.csv --val_manifest '' --batch_size 8 --nGenerate 1

# Visualize scenario test set, model = 20cm IMR
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 206 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/visualize_scenario_test_set.csv --val_manifest '' --batch_size 8 --nGenerate 1

# Visualize scenario test set, model = 20cm Mic
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 207 --ds_rate 1 --nWin 1024 --nFFT 1024 --input_type complex --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/visualize_scenario_test_set.csv --val_manifest '' --batch_size 8 --nGenerate 1


# Visualize scenario test set + ref_dt_compensation, model = REVERB + w_var=100 + IMR
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 187 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/visualize_scenario_test_set_ref_mic_dt_comp.csv --val_manifest '' --batch_size 8 --nGenerate 1

# Visualize scenario test set + ref_dt_compensation, model = REVERB + w_var=100 + Mic
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 186 --ds_rate 1 --nWin 1024 --nFFT 1024 --input_type complex --model_json models/reverb_multimic_6S_realimag.json --mode generate --tr_manifest data_sorted/visualize_scenario_test_set_ref_mic_dt_comp.csv --val_manifest '' --batch_size 8 --nGenerate 1

# Visualize test set, train data = REVERB + visualize_tr, Input = Mic + win/2,  w_var = 0
#CUDA_VISIBLE_DEVICES=0 python3 train.py --model_json models/reverb_multimic_6S_realimag.json --expnum 224 --input_type complex --ds_rate 1 --nWin 1024 --nFFT 1024 --w_var 0 --mode generate --tr_manifest data_sorted/visualize_scenario_augmented_tr_8sample.csv --val_manifest data_sorted/visualize_scenario_augmented_te_8sample.csv --batch_size 8 --nGenerate 1

# Visualize test set, train data = REVERB + visualize_tr, Input = IMR + win/2,  w_var = 0
#CUDA_VISIBLE_DEVICES=0 python3 train.py --model_json models/reverb_multimic_6S_realimag.json --expnum 225 --input_type complex_ratio --ds_rate 8 --nWin 8192 --nFFT 8192 --w_var 0 --mode generate --tr_manifest data_sorted/visualize_scenario_augmented_tr_8sample.csv --val_manifest data_sorted/visualize_scenario_augmented_te_8sample.csv --batch_size 8 --nGenerate 1

# Visualize test set, train data = REVERB + visualize_tr, Input = IMR + shift=2048,  w_var = 0
#CUDA_VISIBLE_DEVICES=0 python3 train.py --model_json models/reverb_multimic_6S_realimag.json --expnum 226 --input_type complex_ratio --ds_rate 8 --nWin 8192 --nFFT 8192 --w_var 0 --hop_length 2048 --mode generate --tr_manifest data_sorted/visualize_scenario_augmented_tr_8sample.csv --val_manifest data_sorted/visualize_scenario_augmented_te_8sample.csv --batch_size 8 --nGenerate 1


# Visualize test set, train data = REVERB + visualize_tr, Input = Mic + win/2,  w_var = 100 (don't have to put w_var at test time)
#CUDA_VISIBLE_DEVICES=0 python3 train.py --model_json models/reverb_multimic_6S_realimag.json --expnum 227 --input_type complex --ds_rate 1 --nWin 1024 --nFFT 1024 --w_var 0 --mode generate --tr_manifest data_sorted/visualize_scenario_augmented_tr_8sample.csv --val_manifest data_sorted/visualize_scenario_augmented_te_8sample.csv --batch_size 8 --nGenerate 1

# Visualize test set, train data = REVERB + visualize_tr, Input = IMR + win/2,  w_var = 100 (don't have to put w_var at test time)
#CUDA_VISIBLE_DEVICES=0 python3 train.py --model_json models/reverb_multimic_6S_realimag.json --expnum 228 --input_type complex_ratio --ds_rate 8 --nWin 8192 --nFFT 8192 --w_var 0 --mode generate --tr_manifest data_sorted/visualize_scenario_augmented_tr_8sample.csv --val_manifest data_sorted/visualize_scenario_augmented_te_8sample.csv --batch_size 8 --nGenerate 1

# Visualize test set, train data = REVERB + visualize_tr, Input = IMR + shift=2048,  w_var = 100 (don't have to put w_var at test time)
#CUDA_VISIBLE_DEVICES=0 python3 train.py --model_json models/reverb_multimic_6S_realimag.json --expnum 229 --input_type complex_ratio --ds_rate 8 --nWin 8192 --nFFT 8192 --w_var 0 --hop_length 2048 --mode generate --tr_manifest data_sorted/visualize_scenario_augmented_tr_8sample.csv --val_manifest data_sorted/visualize_scenario_augmented_te_8sample.csv --batch_size 8 --nGenerate 1


# loss = SD-SDR, tr = DSIR-1cm,  visualize = DSIR-0.1cm, same 5src concat
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 299 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_10S_CW9_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --src_range '4.3,4.6 , 4.3, 4.6, 0, 100' --nMic 2 --loss_type sInvSDR_mag --eval_type srcIndepSDR_mag --eval2_type srcIndepSDR_freqpower --nSource 1 --batch_size 4 --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_0.1cm_RT0.2_nongrid_0.1.csv

# loss = SI-SDR-eq, tr = DSIR-1cm,  visualize = DSIR-0.1cm, same 5src concat
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 297 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_10S_CW9_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --src_range '0, 100, 0, 100, 0, 100' --nMic 2 --loss_type srcIndepSDR_mag --eval_type sInvSDR_mag --eval2_type srcIndepSDR_freqpower --nSource 1 --batch_size 4 --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_0.1cm_RT0.2_nongrid_0.1.csv

# loss = SI-SDR-freq, tr = DSIR-1cm,  visualize = DSIR-0.1cm, same 5src concat
#CUDA_VISIBLE_DEVICES=0 python3 train.py --expnum 298 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_10S_CW9_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --src_range '0, 100, 0, 100, 0, 100' --nMic 2 --loss_type srcIndepSDR_freqpower --eval_type sInvSDR_mag --eval2_type srcIndepSDR_mag --nSource 1 --batch_size 4 --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_0.1cm_RT0.2_nongrid_0.1.csv


# loss = SD-SDR, tr = DSIR-1cm-fixed,  visualize = DSIR-0.1cm-srcfixed, same 5src concat
#CUDA_VISIBLE_DEVICES=0 python3.7 train.py --expnum 314 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_10S_CW9_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --src_range '0, 100, 0, 100, 0, 100' --nMic 2 --loss_type sInvSDR_mag --eval_type srcIndepSDR_mag --eval2_type srcIndepSDR_freqpower --nSource 1 --batch_size 4 --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_0.1cm_nongrid_fixedsrc_2seen1unseen.csv --save_activation False

# loss = SI-SDR-freq, tr = DSIR-1cm-srcfixed, visualize = DSIR-0.1cm-srcfixed, same 5src concat
#CUDA_VISIBLE_DEVICES=0 python3.7 train.py --expnum 315 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_10S_CW9_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --src_range '0, 100, 0, 100, 0, 100' --nMic 2 --loss_type srcIndepSDR_freqpower --eval_type sInvSDR_mag --eval2_type srcIndepSDR_mag --nSource 1 --batch_size 4 --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_0.1cm_nongrid_fixedsrc_2seen1unseen.csv --save_activation False

# loss = SD-SDR
#python3.7 train.py --expnum 320 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_16S_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --nMic 2 --loss_type sInvSDR_mag --eval_type sInvSDR_mag --nSource 1 --batch_size 8 --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_0.1cm_seenSrc.csv --val_manifest data_sorted/L553_fixedmic_hyper_0.1cm_unseenSrc.csv --save_activation False

# loss = SI-SDR, save_activation=False
#python3.7 train.py --expnum 327 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --nMic 2 --loss_type srcIndepSDR_freqpower_diffperT --src_range '0,100,0,100,0,100' --eval_type srcIndepSDR_freqpower_diffperT --nSource 1 --batch_size 8 --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_0.1cm_seenSrc.csv --val_manifest data_sorted/L553_fixedmic_hyper_0.1cm_unseenSrc.csv --save_activation False

# loss = SI-SDR, save_activation=True
#python3.7 train.py --expnum 327 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --nMic 2 --loss_type srcIndepSDR_freqpower_diffperT --src_range '0,100,0,100,0,100' --eval_type srcIndepSDR_freqpower_diffperT --nSource 1 --batch_size 8 --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_0.1cm_seenSrc.csv --val_manifest data_sorted/L553_fixedmic_hyper_0.1cm_unseenSrc.csv --save_activation True

# loss = SI-SDR-by-enhance (331), save_activation=True, nGenerate =1
#python3.7 train.py --expnum 331 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_14S_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --nMic 2 --loss_type srcIndepSDR_freqpower_by_enhanced --src_range '0,100,0,100,0,100' --eval_type sInvSDR_mag --nSource 1 --batch_size 4 --mode generate --nGenerate 1 --tr_manifest data_sorted/L553_fixedmic_hyper_0.1cm_seenSrc.csv --val_manifest data_sorted/L553_fixedmic_hyper_0.1cm_unseenSrc.csv --save_activation True

# loss = SI-SDR-by-Cproj, compare different eps value (340, 341), 
#python3.7 train.py --expnum 341 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_14S_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --nMic 2 --loss_type srcIndepSDR_Cproj_by_SShat --eval_type sInvSDR_mag  --eval2_type srcIndepSDR_Cproj_by_WH --src_range '0,100,0,100,0,100' --nSource 1 --batch_size 8 --mode generate --nGenerate 1 --tr_manifest data_sorted/L553_fixedmic_hyper_1cm_RT0.2_tr.csv --eps 1e-4 --save_activation True

#python3.7 train.py --expnum 341 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_14S_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --nMic 2 --loss_type srcIndepSDR_Cproj_by_SShat --eval_type sInvSDR_mag  --eval2_type srcIndepSDR_Cproj_by_WH --src_range '0,100,0,100,0,100' --nSource 1 --batch_size 8 --mode generate --nGenerate 1 --val_manifest data_sorted/L553_fixedmic_hyper_1cm_RT0.2_tr.csv --eps 1e-12 --save_activation True

#python3.7 train.py --expnum 340 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_14S_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --nMic 2 --loss_type srcIndepSDR_Cproj_by_SShat --eval_type sInvSDR_mag  --eval2_type srcIndepSDR_Cproj_by_WH --src_range '0,100,0,100,0,100' --nSource 1 --batch_size 8 --mode generate --nGenerate 1 --tr_manifest data_sorted/L553_fixedmic_hyper_1cm_RT0.2_tr.csv --eps 1e-12 --save_activation True

#python3.7 train.py --expnum 340 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_14S_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --nMic 2 --loss_type srcIndepSDR_Cproj_by_SShat --eval_type sInvSDR_mag  --eval2_type srcIndepSDR_Cproj_by_WH --src_range '0,100,0,100,0,100' --nSource 1 --batch_size 8 --mode generate --nGenerate 1 --val_manifest data_sorted/L553_fixedmic_hyper_1cm_RT0.2_tr.csv --eps 1e-4 --save_activation True


# C visualize (345)
#python3.7 train.py --expnum 345 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_14S_realimag.json --grid_cm 1 --eval_iter 500 --lR0 1e-4 --nMic 2 --loss_type srcIndepSDR_Cproj_by_SShat --eval_type sInvSDR_mag --src_range '0,100,0,100,0,100' --nSource 1 --eps 1e-12 --clamp_src 8000 --batch_size 8 --mode generate --nGenerate 1 --tr_manifest data_sorted/L553_fixedmic_hyper_1cm_RT0.2_tr.csv


# Metric vs position (363)
#python3.7 train.py --expnum 363 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_14S_ksztime=1_realimag.json --grid_cm 1 --eval_iter 500 --nMic 2 --loss_type srcIndepSDR_Cproj_by_SShat --eval_type sInvSDR_mag --clamp_frame 1 --fix_len_by_cl input --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_1cm_seenSrc_10cm_to_1cm.csv --val_manifest data_sorted/L553_fixedmic_hyper_1cm_unseenSrc_10cm_to_1cm.csv --save_activation True

# Metric vs position (364)
#python3.7 train.py --expnum 364 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_18L_ksztime=1_realimag.json --grid_cm 1 --eval_iter 500 --nMic 2 --loss_type srcIndepSDR_Cproj_by_SShat --eval_type sInvSDR_mag --clamp_frame 1 --fix_len_by_cl input --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_1cm_seenSrc_10cm_to_1cm.csv --val_manifest data_sorted/L553_fixedmic_hyper_1cm_unseenSrc_10cm_to_1cm.csv --save_activation True

# Metric vs position (358)
python3.7 train.py --expnum 358 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_16L_ksztime=1_realimag.json --grid_cm 1 --eval_iter 500 --nMic 2 --loss_type srcIndepSDR_Cproj_by_SShat --eval_type sInvSDR_mag --src_range '0,100,0,100,0,100' --clamp_frame 1 --fix_len_by_cl input --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_1cm_seenSrc_10cm_to_1cm.csv --val_manifest data_sorted/L553_fixedmic_hyper_1cm_unseenSrc_10cm_to_1cm.csv --save_activation True

# Metric vs position (366)
python3.7 train.py --expnum 366 --ds_rate 8 --nWin 8192 --nFFT 8192 --input_type complex_ratio --model_json models/reverb_multimic_18L_ksztime=1_realimag.json --grid_cm 1 --eval_iter 0 --nMic 2 --loss_type srcIndepSDR_Cproj_by_SShat --eval_type sInvSDR_mag --src_range '0,100,0,100,0,100' --clamp_frame 1 --fix_len_by_cl input --mode generate --nGenerate 10000 --tr_manifest data_sorted/L553_fixedmic_hyper_0.1cm_seenSrc_1cm_to_0.1cm.csv --val_manifest data_sorted/L553_fixedmic_hyper_0.1cm_unseenSrc_1cm_to_0.1cm.csv --save_activation True

