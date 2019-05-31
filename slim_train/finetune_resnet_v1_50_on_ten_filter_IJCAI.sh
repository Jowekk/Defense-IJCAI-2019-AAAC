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
PRETRAINED_CHECKPOINT_DIR=/home/yq/Downloads/IJCAI/checkpoints/resnet_v1_50

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=resnet_v1_50

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=./logs/${MODEL_NAME}

# Where the dataset is saved to.
DATASET_DIR=/home/yq/Downloads/IJCAI/

# Fine-tune only the new layers for 1000 steps.
CUDA_VISIBLE_DEVICES="1" python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=ten_filter_IJCAI \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt \
  --checkpoint_exclude_scopes=resnet_v1_50/logits \
#  --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
CUDA_VISIBLE_DEVICES="" python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=ten_filter_IJCAI \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}
