#!/bin/bash

# Fixed Params
PRETRAINED_DATASET="imagenet"
DATASET="places365"
EPSILON=0.03922
LOSS_FN="bounded_logit_fixed_ref"
CONFIDENCE=10
BATCH_SIZE=32
TARGET_CLASS=150
LEARNING_RATE=0.005
NUM_ITERATIONS=2000
WORKERS=4
NGPU=1
SUBF="imagenet_targeted"

TARGET_NETS="alexnet googlenet vgg16 vgg19 resnet152"

for target_net in $TARGET_NETS; do
    python3 train_uap.py \
      --dataset $DATASET \
      --pretrained_dataset $PRETRAINED_DATASET --pretrained_arch $target_net \
      --target_class $TARGET_CLASS --targeted \
      --epsilon $EPSILON \
      --loss_function $LOSS_FN --confidence $CONFIDENCE \
      --num_iterations $NUM_ITERATIONS \
      --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
      --workers $WORKERS --ngpu $NGPU \
      --result_subfolder $SUBF
done
