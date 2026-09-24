#!/bin/bash
# Runs the full experiment: data preprocessing, training and testing the single models,
# and creating and evaluating the ensembles. Run from the root folder of the repository:
#   bash scripts/run.sh

# Make sure the 'python' command on your system represents Python 3
set -e

# Change these values if you want to adjust the training settings or add/remove single models
declare -a networks=( "capsule" "dsp-fwa" "ictu_oculi" "xceptionnet" )
declare -A epochs=( ["capsule"]=25 ["dsp-fwa"]=20 ["ictu_oculi"]=100 ["xceptionnet"]=18 )
declare -A bsize=( ["capsule"]=64 ["dsp-fwa"]=56 ["ictu_oculi"]=40 ["xceptionnet"]=40 )
# Early stopping (number of epochs without improvement of the validation loss), 0 = not used
declare -A estop=( ["capsule"]=0 ["dsp-fwa"]=10 ["ictu_oculi"]=0 ["xceptionnet"]=0 )

# Change these variable values if you want other settings than the default ones
DATA_PATH=$PWD/data/images/
SPLITS_PATH=$PWD/data/splits/
OUTPUT_PATH=$PWD/results/
MODELS_PRETRAINED_PATH=$PWD/models/pre_trained/
MODELS_OUTPUT_PATH=$PWD/models/re_trained/
TRAIN_CSV=train.csv
VAL_CSV=val.csv
TEST_CSV=test.csv

# Data preprocessing
python data/split.py train_csv=$TRAIN_CSV val_csv=$VAL_CSV test_csv=$TEST_CSV

for i in $(seq 1); do # Change $(seq 1) to $(seq 5), $(seq 10), or any number if you want to add more iterations

  # Training single models
  for network in "${networks[@]}"; do
    python train.py with \
    data_path=$DATA_PATH splits_path=$SPLITS_PATH output_path=$OUTPUT_PATH \
    models_pretrained_path=$MODELS_PRETRAINED_PATH models_output_path=$MODELS_OUTPUT_PATH \
    train_csv=$TRAIN_CSV val_csv=$VAL_CSV \
    epochs="${epochs[$network]}" batch_size="${bsize[$network]}" early_stopping="${estop[$network]}" \
    model_name="$network" split_id=$i --name=train_$network
  done

  # Evaluating single models
  for network in "${networks[@]}"; do
    python test.py with \
    data_path=$DATA_PATH splits_path=$SPLITS_PATH output_path=$OUTPUT_PATH \
    models_retrained_path=$MODELS_OUTPUT_PATH test_csv=$TEST_CSV \
    batch_size="${bsize[$network]}" model_name="$network" --name=test_$network
  done
done

# Modeling and evaluating ensembles
python ensemble.py with \
splits_path=$SPLITS_PATH output_path=$OUTPUT_PATH models_saved_path=$MODELS_OUTPUT_PATH \
test_csv=$TEST_CSV --name=ensembles
