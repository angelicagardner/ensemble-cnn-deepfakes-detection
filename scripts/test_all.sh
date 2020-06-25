#!/bin/bash

IMAGES_ROOT=$PWD/data/images/
SPLITS_ROOT=$PWD/data/splits/
RESULTS_ROOT=$PWD/results/models/test/
MODELS_ROOT=$PWD/models/

TEST_CSV=samle.csv

echo ""
echo "-------------------------------------------------"
echo "| Testing Capsule                               |"
echo "-------------------------------------------------"
python3 test.py \
--model_name=capsule --model_path=$MODELS_ROOT \
--images_path=$IMAGES_ROOT \
--csv_path=$SPLITS_ROOT --csv_file=$TEST_CSV \
--output_path=$RESULTS_ROOT

echo ""
echo "-------------------------------------------------"
echo "| Testing DSP-FWA                               |"
echo "-------------------------------------------------"
python3 test.py \
--model_name=dsp-fwa --model_path=$MODELS_ROOT \
--images_path=$IMAGES_ROOT \
--csv_path=$SPLITS_ROOT --csv_file=$TEST_CSV \
--output_path=$RESULTS_ROOT

echo ""
echo "-------------------------------------------------"
echo "| Testing Ictu Oculi                            |"
echo "-------------------------------------------------"
python3 test.py \
--model_name=ictu_oculi --model_path=$MODELS_ROOT \
--images_path=$IMAGES_ROOT \
--csv_path=$SPLITS_ROOT --csv_file=$TEST_CSV \
--output_path=$RESULTS_ROOT

echo ""
echo "-------------------------------------------------"
echo "| Testing ManTra-Net                            |"
echo "-------------------------------------------------"
python3 test.py \
--model_name=mantranet --model_path=$MODELS_ROOT \
--images_path=$IMAGES_ROOT \
--csv_path=$SPLITS_ROOT --csv_file=$TEST_CSV \
--output_path=$RESULTS_ROOT

echo ""
echo "-------------------------------------------------"
echo "| Testing XceptionNet                           |"
echo "-------------------------------------------------"
python3 test.py \
--model_name=xceptionnet --model_path=$MODELS_ROOT \
--images_path=$IMAGES_ROOT \
--csv_path=$SPLITS_ROOT --csv_file=$TEST_CSV \
--output_path=$RESULTS_ROOT