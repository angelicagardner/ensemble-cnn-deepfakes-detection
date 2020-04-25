#!/bin/bash

declare -a networks=("capsule" "dsp-fwa" "xceptionnet")

IMAGES_ROOT=/Users/mnktech/Documents/projects/2dv50e/data/images/
SPLITS_ROOT=/Users/mnktech/Documents/projects/2dv50e/data/splits/

for i in $(seq 3); do
  TRAIN_CSV=train_$i.csv
  VAL_CSV=val_$i.csv
  TEST_CSV=test_$i.csv

  python3 split.py train_csv=$TRAIN_CSV val_csv=$VAL_CSV test_csv=$TEST_CSV

  for network in "${networks[@]}"; do
    python3 test_individual_models.py with \
    data_path=$IMAGES_ROOT splits=$SPLITS_ROOT \
    train_csv=$TRAIN_CSV val_csv=$VAL_CSV test_csv=$TEST_CSV \
    model_name="$network" \
    split_id=$i --name $network
  done
done