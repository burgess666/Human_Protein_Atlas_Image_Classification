#!/bin/sh
#sbatch --job-name=KQ_train --gres=gpu:0 --mem=65536 --cpus-per-task=4 --output=./output/first_train.out launch.sh

python3 data.py