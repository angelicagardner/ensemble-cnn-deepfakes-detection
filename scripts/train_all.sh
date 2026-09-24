#!/bin/bash
# Trains all single models (without running the full experiment). Run from the root folder of the repository:
#   bash scripts/train_all.sh

# Make sure the 'python' command on your system represents Python 3
set -e

declare -a networks=( "capsule" "dsp-fwa" "ictu_oculi" "xceptionnet" )
declare -A epochs=( ["capsule"]=25 ["dsp-fwa"]=20 ["ictu_oculi"]=100 ["xceptionnet"]=18 )
declare -A bsize=( ["capsule"]=64 ["dsp-fwa"]=56 ["ictu_oculi"]=40 ["xceptionnet"]=40 )
declare -A estop=( ["capsule"]=0 ["dsp-fwa"]=10 ["ictu_oculi"]=0 ["xceptionnet"]=0 )

# Change these variable values if you want other settings than the default ones
DATA_PATH=$PWD/data/images/
SPLITS_PATH=$PWD/data/splits/
OUTPUT_PATH=$PWD/results/
MODELS_PRETRAINED_PATH=$PWD/models/pre_trained/
MODELS_OUTPUT_PATH=$PWD/models/re_trained/
TRAIN_CSV=train.csv
VAL_CSV=val.csv

for network in "${networks[@]}"; do
  echo ""
  echo "-------------------------------------------------"
  echo "| Training $network"
  echo "-------------------------------------------------"
  python train.py with \
    data_path=$DATA_PATH splits_path=$SPLITS_PATH output_path=$OUTPUT_PATH \
    models_pretrained_path=$MODELS_PRETRAINED_PATH models_output_path=$MODELS_OUTPUT_PATH \
    train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
    epochs="${epochs[$network]}" batch_size="${bsize[$network]}" early_stopping="${estop[$network]}" \
    model_name="$network" --name=train_$network
done
