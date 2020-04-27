#!/bin/bash

declare -a networks=("resnet50" "vgg16" "vgg19" "xception")

IMAGES_ROOT=/Users/mnktech/Documents/projects/2dv50e/data/images/
SPLITS_ROOT=/Users/mnktech/Documents/projects/2dv50e/data/splits/

TRAIN_CSV=train.csv
VAL_CSV=val.csv
TEST_CSV=test.csv

python3 split.py train_csv=$TRAIN_CSV val_csv=$VAL_CSV test_csv=$TEST_CSV

for network in "${networks[@]}"; do
  python3 individual_models.py with \
  data_path=$IMAGES_ROOT splits_path=$SPLITS_ROOT \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV test_csv=$TEST_CSV \
  epochs=500 model_name="$network" 
done

python3 ensemble.py