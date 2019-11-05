#!/bin/bash

if [ $# -lt 2 ]
then
	echo "Usage : $0 expnum nGenerate"
	exit
fi

case "$1" in
596)	echo "596 X-->S baseline"
	python3.7 train.py --expnum 596 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type S --nWin 1024 --nFFT 1024 --input_type complex --mode generate --nGenerate $2 --save_activation True
	;;
597)	echo "597 X-->W baseline"
	python3.7 train.py --expnum 597 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 1024 --nFFT 1024 --input_type complex --mode generate  --nGenerate $2 --save_activation True
	;;
598) 	echo "598 IMR->W, loss = tarIR"
	python3.7 train.py --expnum 598 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 20 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192  --ds_rate 8 --input_type complex_ratio --mode generate --save_activation True --nGenerate $2
	;;
599)	echo "599 IMR->W, loss = tarIR + 0.1virIR"
	python3.7 train.py --expnum 599 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --loss2_type WH_sum_diff_negative_ref --eval_type SDR_em_mag --eval2_type SDR_C_mag --eval2_type  SDR_Wdiff_realimag_negative_ref --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192  --ds_rate 8 --input_type complex_ratio --w_loss2 0.1 --mode generate --save_activation True --nGenerate $2
	;;

604)	echo "604 X->S (8192)"
	python3.7 train.py --expnum 604 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 20 --match_domain mag --src_dependent True --out_type S --nWin 8192 --nFFT 8192 --input_type complex --mode generate --save_activation True --nGenerate $2
;;

605)	echo "605 X->W (8192)"
	python3.7 train.py --expnum 605 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 20 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex --mode generate --save_activation True --nGenerate $2
;;

6010)   echo "(src free) 6010 RTF --> C"
	python3.7 train.py --expnum 6010 --model_type cMLP --nLayer 3 --nHidden 512 --nFreq 1601 --grid_cm 1 --nMic 2 --loss_type WH_sum_diff_positive_tar --te1_manifest L553_33_0.1_srcfree_RT0.2_ref_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 961 --ds_rate 1 --w_loss2 0 --mode generate --save_activation True --nGenerate $2

# TR/VAL/TE1 manifest 모두 생성하고 싶은것인지 실행 전 항상 확인

esac
