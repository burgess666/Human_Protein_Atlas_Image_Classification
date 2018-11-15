#!/bin/sh
#sbatch --job-name=KQ_human --gres=gpu:1 --mem=4096 --cpus-per-task=2 --nodelist=AIRC-01 launch_tf.sh

python3 train_tf.py