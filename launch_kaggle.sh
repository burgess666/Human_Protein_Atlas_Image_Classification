#!/bin/sh
#sbatch --job-name=KQ_train --gres=gpu:0 --mem=65000 --cpus-per-task=4 --output=./output_kaggle_selu.out launch_kaggle.sh

python3 kaggle_train.py