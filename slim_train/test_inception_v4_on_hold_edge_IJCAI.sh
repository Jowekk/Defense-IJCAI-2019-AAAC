#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an Inception Resnet V2 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_resnet_v2_on_flowers.sh
set -e

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/yq/Downloads/IJCAI/checkpoints/inception_v4

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=inception_v4

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=./logs/hold_edge_inception_v4/

# Where the dataset is saved to.
DATASET_DIR=/home/yq/Downloads/IJCAI/

# Run evaluation.
CUDA_VISIBLE_DEVICES="0" python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --batch_size=32 \
  --dataset_name=hold_edge_IJCAI \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}
