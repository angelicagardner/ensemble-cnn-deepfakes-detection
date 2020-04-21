#!/bin/bash

declare -a networks=("xceptionnet", "meso4")

TRAIN_ROOT=data/videos
VAL_ROOT=data/videos
TEST_ROOT=data/videos

for i in $(seq 5); do
  TRAIN_CSV=train_$i.csv
  VAL_CSV=val_$i.csv
  TEST_CSV=test_$i.csv

  python3 split.py train_csv=$TRAIN_CSV val_csv=$VAL_CSV test_csv=$TEST_CSV

  for network in "${networks[@]}"; do
    python3 train_individual_models.py with \
    train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
    val_root=$VAL_ROOT val_csv=$VAL_CSV \
    test_root=$TEST_ROOT test_csv=$TEST_CSV \
    model_name="$network" \
    split_id=$i --name $network
  done
done