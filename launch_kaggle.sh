#!/bin/sh
#sbatch --job-name=KQ_train --gres=gpu:1 --mem=65000 --cpus-per-task=4 --output=./output_kaggle.out launch_kaggle.sh

python3 kaggle_train.py