#!/bin/bash

set -e

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
#PRETRAINED_CHECKPOINT_DIR=/home/yq/Downloads/IJCAI/checkpoints/alexnet

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=alexnet_v2

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=./logs/${MODEL_NAME}

# Where the dataset is saved to.
DATASET_DIR=/home/yq/Downloads/IJCAI/


# Run evaluation.
CUDA_VISIBLE_DEVICES="" python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=IJCAI \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}
