#!/bin/bash

export CUDA_VISIBLE_DEVICES=""
DATA_DIR="/home/ubuntu/code/A_skip_thoughts_1/skip_thoughts/training_data"

MODEL_DIR="/home/ubuntu/code/A_skip_thoughts_2/skip_thoughts/model"

bazel-bin/skip_thoughts/track_perplexity \
  --input_file_pattern="${DATA_DIR}/validation-?????-of-00001" \
  --checkpoint_dir="${MODEL_DIR}/train" \
  --eval_dir="${MODEL_DIR}/val" \
  --num_eval_examples=5000
