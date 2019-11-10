#!/bin/bash

if [ $# -lt 3 ]
then
	echo "Usage : $0 expnum nGenerate save_activation"
	exit
fi

post_arguments="--mode generate --save_activation $3 --nGenerate $2"  # single quote does not work

case "$1" in
596)	echo "596 X-->S baseline"
	python3.7 train.py --expnum 596 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type S --nWin 1024 --nFFT 1024 --input_type complex --mode generate --nGenerate $2 --save_activation $3
	;;
597)	echo "597 X-->W baseline"
	python3.7 train.py --expnum 597 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 1024 --nFFT 1024 --input_type complex --mode generate  --nGenerate $2 --save_activation $3
	;;
598te1) 	echo "598 IMR->W, loss = tarIR"
	python3.7 train.py --expnum 598 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 20 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192  --ds_rate 8 --input_type complex_ratio --mode generate --save_activation $3 --nGenerate $2
	;;
599te1)	echo "599 IMR->W, loss = tarIR + 0.1virIR"
	python3.7 train.py --expnum 599 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --loss2_type WH_sum_diff_negative_ref --eval_type SDR_em_mag --eval2_type SDR_C_mag --eval2_type  SDR_Wdiff_realimag_negative_ref --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192  --ds_rate 8 --input_type complex_ratio --w_loss2 0.1 --mode generate --save_activation $3--nGenerate $2
	;;

604te1)	echo "604 X->S (8192)"
	python3.7 train.py --expnum 604 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 20 --match_domain mag --src_dependent True --out_type S --nWin 8192 --nFFT 8192 --input_type complex --mode generate --save_activation $3 --nGenerate $2
;;

604te2)	echo "604 X->S (8192)  (te2: [4.3, 4.6], 1cm, #src=1)"
	python3.7 train.py --expnum 604 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 20 --match_domain mag --src_dependent True --out_type S --nWin 8192 --nFFT 8192 --input_type complex --mode generate --save_activation $3 --nGenerate $2
;;

605)	echo "605 X->W (8192)"
	python3.7 train.py --expnum 605 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 20 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex --mode generate --save_activation $3 --nGenerate $2
;;

605te2)	echo "605 X->W (8192)  (te2: [4.3, 4.6], 1cm, #src=1"
	python3.7 train.py --expnum 605 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 20 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex --mode generate --save_activation $3 --nGenerate $2
;;


6010te1)   echo "(src free) 6010 RTF --> C"
	python3.7 train.py --expnum 6010 --model_type cMLP --nLayer 3 --nHidden 512 --nFreq 1601 --grid_cm 1 --nMic 2 --loss_type WH_sum_diff_positive_tar --te1_manifest L553_33_0.1_srcfree_RT0.2_ref_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 961 --ds_rate 1 --w_loss2 0 --mode generate --save_activation $3 --nGenerate $2
;;

610te1)   echo "610 IMR->W (ksztime > 1)"
	python3.7 train.py --expnum 610 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 8 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio --mode generate --save_activation $3 --nGenerate $2
;;

612te1)   echo "612 IMR->W (ksztime = 1)"
	python3.7 train.py --expnum 612 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 8 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio --mode generate --save_activation $3 --nGenerate $2
;;

618te1)	echo "618 tarIR, 1cm, ksztime = 1"
	python3.7 train.py --expnum 618 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio --interval_cm_tr 1 --mode generate --save_activation $3 --nGenerate $2
;;

618te2)	echo "618 tarIR, 1cm, ksztime = 1 (te2: [4.3, 4.6], 1cm, #src=1)"
	python3.7 train.py --expnum 618 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio --interval_cm_tr 1 --mode generate --save_activation $3 --nGenerate $2
;;

621te1)   echo "621 tarIR+0.1refIR, 1cm, ksztime = 1"
	python3.7 train.py --expnum 621 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --loss2_type WH_sum_diff_negative_ref --eval_type SDR_em_mag --eval2_type SDR_C_mag --te1_manifest L553_33_0.1_nSrc_5_RT0.2_ref_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio --interval_cm_tr 1 --mode generate --save_activation $3 --nGenerate $2
;;

621te2)   echo "621 tarIR+0.1refIR, 1cm, ksztime = 1 (te2: [4.3, 4.6], 1cm, #src=1)"
	python3.7 train.py --expnum 621 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --loss2_type WH_sum_diff_negative_ref --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio --interval_cm_tr 1 --mode generate --save_activation $3 --nGenerate $2
;;

598te2) 	echo "598 IMR->W, loss = tarIR (te2: [4.3, 4.6], 1cm, #src=1)"
		python3.7 train.py --expnum 598 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 20 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192  --ds_rate 8 --input_type complex_ratio --mode generate --save_activation $3 --nGenerate $2
;;

599te2) 	echo "599 IMR->W, loss = tarIR + 0.1virIR (te2: [4.3, 4.6], 1cm, #src=1)"
	python3.7 train.py --expnum 599 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --loss2_type WH_sum_diff_negative_ref --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192  --ds_rate 8 --input_type complex_ratio --w_loss2 0.1 --mode generate --save_activation $3--nGenerate $2
;;

621te2)		echo "621 tarIR+0.1refIR, 1cm, ksztime = 1  (te2: [4.3, 4.6], 1cm, #src=1)"
	python3.7 train.py --expnum 621 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --loss2_type WH_sum_diff_negative_ref --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio --interval_cm_tr 1 --mode generate --save_activation $3 --nGenerate $2
;;


616te2)		echo "616 tarIR, 5cm (te2: [4.3, 4.6], 1cm, #src=1)"
	python3.7 train.py --expnum 616 --model_type unet --model_json models/reverb_multimic_8S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

622te2)		echo "622 tarIR + 0.1refIR, 5cm (te2: [4.3, 4.6], 1cm, #src=1)"
	python3.7 train.py --expnum 622 --model_type unet --model_json models/reverb_multimic_8S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --loss2_type WH_sum_diff_negative_ref --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

614te2)		echo "614 X->S, 5cm (te2: [4.3, 4.6], 1cm, #src=1)"
	python3.7 train.py --expnum 614 --model_type unet --model_json models/reverb_multimic_8S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type S --nWin 8192 --nFFT 8192 --input_type complex $post_arguments

;;

615te2)		echo "615 X->W, 5cm (te2: [4.3, 4.6], 1cm, #src=1)"
	python3.7 train.py --expnum 615 --model_type unet --model_json models/reverb_multimic_8S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
;;

# TR/VAL/TE1 manifest ��� �����ϰ� ���������� ���� �� �׻� Ȯ��

esac