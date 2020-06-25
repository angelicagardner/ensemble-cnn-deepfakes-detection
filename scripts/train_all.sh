#!/bin/bash

IMAGES_ROOT=$PWD/data/images/
SPLITS_ROOT=$PWD/data/splits/
RESULTS_ROOT=$PWD/results/
MODELS_ROOT=$PWD/models/

TRAIN_CSV=train.csv
VAL_CSV=val.csv

echo ""
echo "-------------------------------------------------"
echo "| Training Capsule                              |"
echo "-------------------------------------------------"
python3 train.py with \
  data_path=$IMAGES_ROOT splits_path=$SPLITS_ROOT \
  results_path=$RESULTS_ROOT models_path=$MODELS_ROOT \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
  epochs=25 batch_size=64 \
  model_name=capsule --name capsule

echo ""
echo "-------------------------------------------------"
echo "| Training DSP-FWA                              |"
echo "-------------------------------------------------"
python3 train.py with \
  data_path=$IMAGES_ROOT splits_path=$SPLITS_ROOT \
  results_path=$RESULTS_ROOT models_path=$MODELS_ROOT \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
  epochs=20 batch_size=56 \
  model_name=dsp-fwa --name dsp-fwa

echo ""
echo "-------------------------------------------------"
echo "| Training Ictu Oculi                           |"
echo "-------------------------------------------------"
python3 train.py with \
  data_path=$IMAGES_ROOT splits_path=$SPLITS_ROOT \
  results_path=$RESULTS_ROOT models_path=$MODELS_ROOT \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
  epochs=100 batch_size=40 \
  model_name=ictu_oculi --name ictu_oculi

echo ""
echo "-------------------------------------------------"
echo "| Training ManTra-Net                           |"
echo "-------------------------------------------------"
python3 train.py with \
  data_path=$IMAGES_ROOT splits_path=$SPLITS_ROOT \
  results_path=$RESULTS_ROOT models_path=$MODELS_ROOT \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
  epochs=500 batch_size=24 \
  model_name=mantranet --name mantranet

echo ""
echo "-------------------------------------------------"
echo "| Training XceptionNet                          |"
echo "-------------------------------------------------"
python3 train.py with \
  data_path=$IMAGES_ROOT splits_path=$SPLITS_ROOT \
  results_path=$RESULTS_ROOT models_path=$MODELS_ROOT \
  train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
  epochs=18 batch_size=40 \
  model_name=xceptionnet --name xceptionnet