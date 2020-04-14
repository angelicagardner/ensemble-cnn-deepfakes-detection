#!/bin/bash

declare -a nets=("xceptionnet")

TRAIN_ROOT=data/videos
VAL_ROOT=data/videos
TEST_ROOT=data/videos

for i in $(seq 5); do
  TRAIN_CSV=data/splits/train_$i.csv
  VAL_CSV=data/splits/val_$i.csv
  TEST_CSV=data/splits/test_$i.csv

  for net in "${nets[@]}"; do
    python3 train.py with train_root=$TRAIN_ROOT val_root=$VAL_ROOT \
    test_root=$TEST_ROOT model_name="$net"
  done
done