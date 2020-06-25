#!/bin/bash

declare -a networks=( "capsule" "dsp-fwa" "ictu_oculi" "mantranet" "xceptionnet" )
declare -a epochs=( ["capsule"]=25 ["dsp-fwa"]=20 ["ictu_oculi"]=100 \
                  ["mantranet"]=500 ["xceptionnet"]=18 )
declare -a bsize=( ["capsule"]=64 ["dsp-fwa"]=56 ["ictu_oculi"]=40 \
                   ["mantranet"]=24 ["xceptionnet"]=40 )

IMAGES_ROOT=$PWD/data/images/
SPLITS_ROOT=$PWD/data/splits/
RESULTS_ROOT=$PWD/results/
MODELS_ROOT=$PWD/models/

TRAIN_CSV=train.csv
VAL_CSV=val.csv
TEST_CSV=test.csv

# Data preprocessing
python3 split.py train_csv=$TRAIN_CSV val_csv=$VAL_CSV test_csv=$TEST_CSV

# Training and testing individual models
for network in "${networks[@]}"; do
  python3 individual_models.py with \
  data_path=$IMAGES_ROOT splits_path=$SPLITS_ROOT \
  results_path=$RESULTS_ROOT models_path=$MODELS_ROOT \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV test_csv=$TEST_CSV \
  epochs="${epochs[$network]}" batch_size="${bsize[$network]}" \
  model_name="$network" split_id=1 --name $network
done

# Ensemble modelling
python3 ensemble.py