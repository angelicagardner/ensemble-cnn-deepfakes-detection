#!/bin/bash
# Tests all single models on the test set (without running the full experiment). Run from the root folder of the repository:
#   bash scripts/test_all.sh

# Make sure the 'python' command on your system represents Python 3
set -e

declare -a networks=( "capsule" "dsp-fwa" "ictu_oculi" "xceptionnet" )

# Change these variable values if you want other settings than the default ones
DATA_PATH=$PWD/data/images/
SPLITS_PATH=$PWD/data/splits/
OUTPUT_PATH=$PWD/results/
MODELS_OUTPUT_PATH=$PWD/models/re_trained/
TEST_CSV=test.csv

for network in "${networks[@]}"; do
  echo ""
  echo "-------------------------------------------------"
  echo "| Testing $network"
  echo "-------------------------------------------------"
  python test.py with \
    data_path=$DATA_PATH splits_path=$SPLITS_PATH output_path=$OUTPUT_PATH \
    models_retrained_path=$MODELS_OUTPUT_PATH test_csv=$TEST_CSV \
    model_name="$network" --name=test_$network
done
