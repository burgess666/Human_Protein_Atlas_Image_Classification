#!/bin/sh
#sbatch --job-name=KQ_human --gres=gpu:0 --mem=65000 --cpus-per-task=2 --output=output_train launch_tf.sh

python3 train_tf.py