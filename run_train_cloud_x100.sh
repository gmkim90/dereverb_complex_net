#!/bin/bash

# cloud

arguments=$1
for ((epoch=0;epoch<=1000;epoch++)); do
    echo "training $epoch epoch"

    python3.7 train.py $arguments --start_epoch $epoch --num_epochs $((epoch+1)) --start_ratio 0.0 --end_ratio 0.5

    python3.7 train.py $arguments --start_epoch $epoch --num_epochs $((epoch+1)) --start_ratio 0.5 --end_ratio 1.0
    sleep 5

done
