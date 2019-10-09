#!/bin/bash

# cloud

arguments=$1
for ((epoch=0;epoch<=10000;epoch++)); do
    echo "training $epoch epoch"

    python3.7 train.py $arguments --start_epoch $epoch --num_epochs $((epoch+1)) --start_ratio 0.0 --end_ratio 0.25
   sleep 5

    if [ $epoch == 0 ]; then # git upload after successful execution
	git checkout cloud
	git add .gitignore
	git add *.py
	git add *.sh
	git add models
	git add data_sorted/*.py
        arguments_token=($arguments)
	git commit -m "${arguments_token[0]} ${arguments_token[1]}"   	
	git push origin cloud
    fi

    python3.7 train.py $arguments --start_epoch $epoch --num_epochs $((epoch+1)) --start_ratio 0.25 --end_ratio 0.5
    sleep 5

    python3.7 train.py $arguments --start_epoch $epoch --num_epochs $((epoch+1)) --start_ratio 0.5 --end_ratio 0.75
    sleep 5

    python3.7 train.py $arguments --start_epoch $epoch --num_epochs $((epoch+1)) --start_ratio 0.75 --end_ratio 1.0
    sleep 5

done
