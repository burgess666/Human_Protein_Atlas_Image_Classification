#!/bin/sh
#sbatch --job-name=KQ_submit --gres=gpu:0 --mem=16000 --cpus-per-task=4 --output=output_submit.out submit.sh

python3 submit.py