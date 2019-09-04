#!/bin/bash

#arguments=$2
arguments=$1
#echo "!!!!!!!!!!!!!! train only 1 epoch for DEBUG !!!!!!!!!!!!!!!!!!!!!"
for ((epoch=0;epoch<=100;epoch++)); do
    echo "training $epoch epoch"

    python3.7 train.py $arguments --start_epoch $epoch --num_epochs $((epoch+1))
    sleep 5

done
