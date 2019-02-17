#!/bin/bash
# This script is used for training, inference and benchmarking
# the Adaptive Affinity Fields method with PSPNet on PASCAL VOC
# 2012. Users could also modify from this script for their use
# case.
#
# Usage:
#   # From Adaptive_Affinity_Fields/ directory.
#   bash bashscripts/voc12/train_pspnet_aaf.sh
#
#

# Set up parameters for training.
BATCH_SIZE=10
TRAIN_INPUT_SIZE=240,240
WEIGHT_DECAY=5e-4
ITER_SIZE=1
NUM_STEPS=30001
NUM_CLASSES=3
KLD_MARGIN=3.0
KLD_LAMBDA_1=1.0
KLD_LAMBDA_2=1.0

# Set up parameters for inference.
INFERENCE_INPUT_SIZE=528,528
INFERENCE_STRIDES=512,512
INFERENCE_SPLIT=val

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/

# Set up the procedure pipeline.
IS_TRAIN_1=1
IS_INFERENCE_1=1
IS_BENCHMARK_1=0
IS_TRAIN_2=1
IS_INFERENCE_2=1
IS_BENCHMARK_2=0

# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Set up the data directory.
DATAROOT=/home/data1/U-net/H-DenseUNet/data/

# Train for the 1st stage.
if [ ${IS_TRAIN_1} -eq 1 ]; then
  python pyscripts/train/train_aaf.py\
    --snapshot-dir ${SNAPSHOT_DIR}/stage1\
    --restore-from resnet_v1_101.ckpt\
    --data-list dataset/voc12/train+.txt\
    --data-dir ${DATAROOT}\
    --batch-size ${BATCH_SIZE}\
    --save-pred-every ${NUM_STEPS}\
    --update-tb-every 50\
    --input-size ${TRAIN_INPUT_SIZE}\
    --learning-rate 1e-3\
    --weight-decay ${WEIGHT_DECAY}\
    --iter-size ${ITER_SIZE}\
    --num-classes ${NUM_CLASSES}\
    --num-steps $(($NUM_STEPS+1))\
    --random-mirror\
    --random-scale\
    --random-crop\
    --kld-margin ${KLD_MARGIN}\
    --kld-lambda-1 ${KLD_LAMBDA_1}\
    --kld-lambda-2 ${KLD_LAMBDA_2}\
    --is-training
fi

# Inference for the 1st stage.
if [ ${IS_INFERENCE_1} -eq 1 ]; then
  python pyscripts/inference/inference.py\
    --input-size ${INFERENCE_INPUT_SIZE}\
    --strides ${INFERENCE_STRIDES}\
    --restore-from ${SNAPSHOT_DIR}/stage1/model.ckpt-${NUM_STEPS}\
    --num-classes ${NUM_CLASSES}\
    --ignore-label 255\
    --save_path ${SNAPSHOT_DIR}/stage1/results/
fi

# Benchmark for the 1st stage.
if [ ${IS_BENCHMARK_1} -eq 1 ]; then
  python pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred-dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}/gray/\
    --gt-dir ${DATAROOT}/VOCdevkit/VOC2012/segcls/\
    --num-classes ${NUM_CLASSES}
fi

# Train for the 2nd stage.
if [ ${IS_TRAIN_2} -eq 1 ]; then
  python pyscripts/train/train_aaf.py\
    --snapshot-dir ${SNAPSHOT_DIR}/stage2\
    --restore-from ${SNAPSHOT_DIR}/stage1/model.ckpt-${NUM_STEPS}\
    --data-list dataset/voc12/train.txt\
    --data-dir ${DATAROOT}\
    --batch-size ${BATCH_SIZE}\
    --save-pred-every ${NUM_STEPS}\
    --update-tb-every 50\
    --input-size ${TRAIN_INPUT_SIZE}\
    --learning-rate 1e-4\
    --weight-decay ${WEIGHT_DECAY}\
    --iter-size ${ITER_SIZE}\
    --num-classes ${NUM_CLASSES}\
    --num-steps $(($NUM_STEPS+1))\
    --random-mirror\
    --random-scale\
    --random-crop\
    --kld-margin ${KLD_MARGIN}\
    --kld-lambda-1 ${KLD_LAMBDA_1}\
    --kld-lambda-2 ${KLD_LAMBDA_2}\
    --is-training
fi

# Inference for the 2nd stage.
if [ ${IS_INFERENCE_2} -eq 1 ]; then
  python pyscripts/inference/inference.py\
    --input-size ${INFERENCE_INPUT_SIZE}\
    --strides ${INFERENCE_STRIDES}\
    --restore-from ${SNAPSHOT_DIR}/stage2/model.ckpt-${NUM_STEPS}\
    --num-classes ${NUM_CLASSES}\
    --ignore-label 255\
    --save_path ${SNAPSHOT_DIR}/stage2/results/
fi

# Benchmark for the 2nd stage.
if [ ${IS_BENCHMARK_2} -eq 1 ]; then
  python pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred-dir ${SNAPSHOT_DIR}/stage2/results/${INFERENCE_SPLIT}/gray/\
    --gt-dir ${DATAROOT}/VOCdevkit/VOC2012/segcls/\
    --num-classes ${NUM_CLASSES}
fi
