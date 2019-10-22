#!/bin/bash

# brain21

arguments=$1
#for ((epoch=0;epoch<=10000;epoch++)); do
for epoch in $(seq 0 100 1000000); do
    echo "training $epoch epoch"

    python3.7 train.py $arguments --start_epoch $epoch --num_epochs $((epoch+100)) --start_ratio 0 --end_ratio 1
    sleep 5

    if [ $epoch == 0 ]; then # git upload after successful execution
	git checkout cloud_srcfree
	git add *.py
	git add *.sh
	git add models
        arguments_token=($arguments)
	git commit -m "${arguments_token[0]} ${arguments_token[1]}"   	
	git push origin cloud_srcfree
    fi

done
