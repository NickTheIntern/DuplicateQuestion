#!/bin/bash

DATA_DIR="/home/ubuntu/code/A_skip_thoughts_1/skip_thoughts/training_data"

MODEL_DIR="/home/ubuntu/code/A_skip_thoughts_2/skip_thoughts/model"

bazel build -c opt skip_thoughts/...

sudo bazel-bin/skip_thoughts/train \
  --input_file_pattern="${DATA_DIR}/train-?????-of-00100" \
  --train_dir="${MODEL_DIR}/train"
