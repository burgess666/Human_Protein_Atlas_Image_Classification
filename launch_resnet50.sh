#!/bin/sh
#sbatch --job-name=KQ_resnet50 --gres=gpu:1 --mem=4096 --cpus-per-task=2 --output=output_train_resnet50.out launch_resnet50.sh

python3 train_resnet50.py