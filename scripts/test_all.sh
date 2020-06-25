#!/bin/bash

# Make sure the 'python' commmand on your system represents Python 3

# Change these variable values if you want other settings than the default ones
DATA_PATH=$PWD/data/images/
SPLITS_PATH=$PWD/data/splits/
OUTPUT_PATH=$PWD/results/
MODELS_PATH=$PWD/models/
MODELS_OUTPUT_PATH=$PWD/models/re_trained/
TEST_CSV=test.csv

echo ""
echo "-------------------------------------------------"
echo "| Testing Capsule                               |"
echo "-------------------------------------------------"
python test.py \
--model_name=capsule --model_path=$MODELS_PATH \
--data_path=$DATA_PATH \
--splits_path=$SPLITS_ROOT --test_csv=$TEST_CSV \
--output_path=$RESULTS_ROOT

echo ""
echo "-------------------------------------------------"
echo "| Testing DSP-FWA                               |"
echo "-------------------------------------------------"
python test.py \
--model_name=dsp-fwa --model_path=$MODELS_PATH \
--data_path=$DATA_PATH  \
--splits_path=$SPLITS_ROOT --test_csv=$TEST_CSV \
--output_path=$RESULTS_ROOT

echo ""
echo "-------------------------------------------------"
echo "| Testing Ictu Oculi                            |"
echo "-------------------------------------------------"
python test.py \
--model_name=ictu_oculi --model_path=$MODELS_PATH \
--data_path=$DATA_PATH  \
--splits_path=$SPLITS_ROOT --test_csv=$TEST_CSV \
--output_path=$RESULTS_ROOT

echo ""
echo "-------------------------------------------------"
echo "| Testing XceptionNet                           |"
echo "-------------------------------------------------"
python test.py \
--model_name=xceptionnet --model_path=$MODELS_PATH \
--data_path=$DATA_PATH \
--splits_path=$SPLITS_ROOT --test_csv=$TEST_CSV \
--output_path=$RESULTS_ROOT