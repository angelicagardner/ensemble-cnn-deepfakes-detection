# 2dv50e

This project contains the source code of all experiments described in 'Degree Project at Bachelor Level - VT2020 - Linnaeus University.'

This repository is currently under mantainance, feel free to open an issue if you encounter problems or something that needs fixing.


## Abstract

> Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
> Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
> Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
> Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.


## Project structure

- Datasets folder for placing all videos.
```
    ./data/videos
```

- Datasets folder where all image frames are placed when splitting the dataset.
```
    ./data/images
```

- Splits folder where information about subsets for train, validation, and test splits will be stored.
```
    ./data/splits
```

- Models folder where you find individual model implementations.
```
    ./models/<model_name>
```


## Setup

Run `pip3 install -r requirements.txt`


### Requirements

See file *requirements.txt* for necessary packages.

### Configurations

Change configurations from default ones in ```run.sh``` or directly through the terminal by adding the following commands:
```shell
python3 individual_models.py with
data_path=<path to video frames (folder containing images)>
splits_path=<path to CSV files with information about train, validation, and test splits>
results_path=<path to output folder, will contain evaluation results>
models_path=<path to saved models>
train_csv=<path to train set CSV>
val_csv=<path to validation set CSV>
test_csv=<path to test set CSV>
epochs=<number of times a model will go through the complete training set>
batch_size=<the amount of data examples included in each epoch>
early_stopping=<training is stopped early if the validation loss has not decrease further after this number of epochs>
model_name=<CNN model>
``` 


## Datasets

1. Download the datasets.
2. Put all videos into the same directory as a CSV-file with information about each video (e.g. `./data/videos`). Alternately, take a look at the file `data_sorting.py` for the code used in this research to organise the data.

### Celeb-DF
The dataset can be downloaded [here](https://github.com/danmohaha/celeb-deepfakeforensics#download).

### DeepFakeDetection
The dataset can be downloaded together with the regular FaceForensics++ dataset found [here](https://github.com/ondyari/FaceForensics/#access).

### Deepfake Detection Challenge
The small sample training set was used during this experiment. There's a much larger full training set also available that can be used to replace this smaller sample set.
Both the small sample training set and the full training set can be downloaded [here](https://www.kaggle.com/c/deepfake-detection-challenge/data).


## Individual models

### (1) Capsule
### (2) DSP-FWA
### (3) Ictu Oculi
### (4) ManTra-Net
### (5) XceptionNet


## Ensemble

...


## References
- Xception PyTorch


## Citation

This research was carried out while the author studied at Linneaus University, Sweden.

If you use this code or research as a reference, please cite:
```
...
```
or reference (IEEE):
```
...
```