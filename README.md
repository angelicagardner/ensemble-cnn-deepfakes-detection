# 2dv50e

This project contains the source code of all experiments described in 'Degree Project at Bachelor Level - VT2020 - Linnaeus University.'


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

- Splits folder where information about subsets for train, validation, and test splits will be stored.
```
    ./data/splits
```

- Models folder where you find individual model implementations.
```
    ./models/<model_name>
```

- Ensemble folder where you find the ensemble models.
```
    ./ensembles
```


## Setup

Run `pip3 install -r requirements.txt`


### Requirements

See file *requirements.txt* for all needed packages


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

### (1)
### (2)
### (3)


## Ensembles

...


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


#### Notice
This repository is provided "AS IS". The authors and contributors are not responsible for any subsequent usage of this code.