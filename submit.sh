#!/bin/sh
#sbatch --job-name=KQ_submit --gres=gpu:1 --mem=8000 --cpus-per-task=4 --output=output_submit.out --modelist=AIRC-01 submit.sh

python3 submit.py
