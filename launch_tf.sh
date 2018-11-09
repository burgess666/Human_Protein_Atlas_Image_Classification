#!/bin/sh
#sbatch --job-name=KQ_tftrain --gres=gpu:1 --mem=65536 --cpus-per-task=4 launch_tf.sh

python3 train_tf.py