#!/bin/bash

# Make sure the 'python' commmand on your system represents Python 3

# Change these variable values if you want other settings than the default ones
DATA_PATH=$PWD/data/images/
SPLITS_PATH=$PWD/data/splits/
OUTPUT_PATH=$PWD/results/
MODELS_PATH=$PWD/models/
MODELS_PRETRAINED_PATH=$PWD/models/pre_trained/
MODELS_OUTPUT_PATH=$PWD/models/re_trained/
TRAIN_CSV=train.csv
VAL_CSV=val.csv

echo ""
echo "-------------------------------------------------"
echo "| Training Capsule                              |"
echo "-------------------------------------------------"
python train.py with \
  data_path=$DATA_PATH splits_path=$SPLITS_ROOT \
  results_path=$OUTPUT_PATH models_path=$MODELS_PATH \
  models_pretrained_path=$MODELS_PRETRAINED_PATH models_output_path=$MODELS_OUTPUT_PATH \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
  epochs=25 batch_size=64 \
  model_name=capsule --name capsule

echo ""
echo "-------------------------------------------------"
echo "| Training DSP-FWA                              |"
echo "-------------------------------------------------"
python train.py with \
  data_path=$DATA_PATH splits_path=$SPLITS_ROOT \
  results_path=$OUTPUT_PATH models_path=$MODELS_PATH \
  models_pretrained_path=$MODELS_PRETRAINED_PATH models_output_path=$MODELS_OUTPUT_PATH \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
  epochs=20 batch_size=56 \
  model_name=dsp-fwa --name dsp-fwa

echo ""
echo "-------------------------------------------------"
echo "| Training Ictu Oculi                           |"
echo "-------------------------------------------------"
python train.py with \
  data_path=$DATA_PATH splits_path=$SPLITS_ROOT \
  results_path=$OUTPUT_PATH models_path=$MODELS_PATH \
  models_pretrained_path=$MODELS_PRETRAINED_PATH models_output_path=$MODELS_OUTPUT_PATH \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
  epochs=100 batch_size=40 \
  model_name=ictu_oculi --name ictu_oculi

echo ""
echo "-------------------------------------------------"
echo "| Training XceptionNet                          |"
echo "-------------------------------------------------"
python train.py with \
  data_path=$DATA_PATH splits_path=$SPLITS_ROOT \
  results_path=$OUTPUT_PATH models_path=$MODELS_PATH \
  models_pretrained_path=$MODELS_PRETRAINED_PATH models_output_path=$MODELS_OUTPUT_PATH \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
  epochs=18 batch_size=40 \
  model_name=xceptionnet --name xceptionnet