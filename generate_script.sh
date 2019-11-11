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
	python3.7 train.py --expnum 604 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type S --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
;;

605)	echo "605 X->W (8192)"
	python3.7 train.py --expnum 605 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 20 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
;;

605te2)	echo "605 X->W (8192)  (te2: [4.3, 4.6], 1cm, #src=1"
	python3.7 train.py --expnum 605 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
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

622te2)		echo "622 tarIR + 0.1refIR, 5cm (te2: [4.3, 4.6], 1cm, #src=5(<-1))"
	python3.7 train.py --expnum 622 --model_type unet --model_json models/reverb_multimic_8S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

614te2)		echo "614 X->S, 5cm (te2: [4.3, 4.6], 1cm, #src=5(<-1))"
	python3.7 train.py --expnum 614 --model_type unet --model_json models/reverb_multimic_8S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type S --nWin 8192 --nFFT 8192 --input_type complex $post_arguments

;;

615te2)		echo "615 X->W, 5cm (te2: [4.3, 4.6], 1cm, #src=5(<-1))"
	python3.7 train.py --expnum 615 --model_type unet --model_json models/reverb_multimic_8S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --eval2_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
;;


6220te2)	echo "6220 IMR->W, L = Wdiff_gtnormalized 5cm (te2: [4.3, 4.6], 1cm, #src=1))"
		python3.7 train.py --expnum 6220 --model_type unet --model_json models/reverb_multimic_8S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type Wdiff_realimag_gtnormalized --eval2_type SDR_Wdiff_realimag --te2_manifest L553_30301_1_unseenSrc1_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

629te2)		echo "629 IMW->W, L = Cdistortion_pos 5cm (te2: [4.3, 4.6], 1cm, #src=5)"
		python3.7 train.py --expnum 629 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type Cdistortion_pos --eval_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv  --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
;;

6310te2)	echo "6310 IMR->W, L = tarIR 10cm (te2, #src=5)"
		python3.7 train.py --expnum 6310 --model_type unet --model_json models/reverb_multimic_6S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

6320te2)	echo "6320 IMR->W, L = tarIR + 0.1refIR (te2, #src=5)"
		python3.7 train.py --expnum 6320 --model_type unet --model_json models/reverb_multimic_6S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

618te2)	echo "618 tarIR, 1cm, ksztime = 1 (te2, #src=5)"
	python3.7 train.py --expnum 618 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

627te2) echo "627 IMR->W, L = Cdistortion_pos (te2, #src=5)"
	python3.7 train.py --expnum 627 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type Cdistortion_pos --eval_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

621te2) echo "621 IMR->W, L = tarIR + 0.1refIR (te2, #src=5)"
	python3.7 train.py --expnum 621 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --loss2_type WH_sum_diff_negative_ref --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

628te2) echo "628 IMR->W, L = Cdistortion_pos + 0.1Cdistortion_neg"
	python3.7 train.py --expnum 628 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type Cdistortion_pos --eval_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

629te2) echo "629 X->W, L = Cdistortion_pos (te2, #src=5)"
	python3.7 train.py --expnum 629 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type Cdistortion_pos --eval_type SDR_C_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --val_manifest L553_pos100_nSrc_5_30301_val3_RT0.2_ref_ofa.csv  --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
;;


620te2) echo "620 RTF->C, L = WH_sum_diff_positive_tar (te2, #src=5)"
	python3.7 train.py --expnum 620 --model_type cMLP --nLayer 3 --nHidden 128 --nFreq 1601 --grid_cm 1 --nMic 2 --loss_type WH_sum_diff_positive_tar --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --ds_rate 1 $post_arguments
;;

623te2) echo "623 IMR->W, L=tarIR, #src=1 (te2, #src=5)"
	python3.7 train.py --expnum 623 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

624te2) echo "624 IMR->W, L = tarIR + 0.1refIR, #src=1 (te2, #src=5)"
	python3.7 train.py --expnum 624 --model_type unet --model_json models/reverb_multimic_12S_ksztime=1_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --ds_rate 8 --input_type complex_ratio $post_arguments
;;

625te2) echo "625 X->W, L = tarIR, #src=1 (te2, #src=5)"
	python3.7 train.py --expnum 625 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
;;

626te2) echo "626 X->S, L = tarIR, #src=1 (te2, #src=5)"
	python3.7 train.py --expnum 626 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR True --use_ref_IR_te True --batch_size 16 --match_domain mag --src_dependent True --out_type S --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
;;

608te2) echo "608 X->S, L = tarIR, 10cm (te2, #src=5)"
	python3.7 train.py --expnum 608 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv  --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type S --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
;;

609te2) echo "609 X->S, L = tarIR, 10cm (te2, #src=5)"
	python3.7 train.py --expnum 609 --model_type unet --model_json models/reverb_multimic_12S_realimag.json --grid_cm 1 --nMic 2 --loss_type diff --eval_type SDR_em_mag --te2_manifest L553_30301_1_unseenSrc5_ref1_ofa.csv --use_ref_IR False --use_ref_IR_te False --batch_size 16 --match_domain mag --src_dependent True --out_type W --nWin 8192 --nFFT 8192 --input_type complex $post_arguments
;;





# TR/VAL/TE1 manifest 모두 생성하고 싶은것인지 실행 전 항상 확인

esac
