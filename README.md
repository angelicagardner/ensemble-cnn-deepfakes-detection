# Stronger Together? An Ensemble of CNNs for Deepfakes Detection.

This project contains the source code of the experiment described in *'Stronger Together? An Ensemble of CNNs for Deepfakes Detection'* which was my degree project at bachelor level at Linnaeus University during spring semester 2020. As the class has ended, this project is currently not active and I consider this repository as under mantainance. However, do feel free to open an issue if you encounter problems or something that needs fixing.


## Abstract

> Deepfakes technology is a face swap technique that enables anyone to replace faces in a video, with highly realistic results. Despite its usefulness, if used maliciously, this technique can have a significant impact on society, for instance, through the spreading of fake news or cyberbullying. This makes the ability of deepfakes detection a problem of utmost importance. In this paper, I tackle the problem of deepfakes detection by identifying deepfakes forgeries in video sequences. Inspired by the state-of-the-art, I study the ensembling of different machine learning solutions built on convolutional neural networks (CNNs) and use these models as objects for comparison between ensemble and single model performances. Existing work in the research field of deepfakes detection suggests that escalated challenges posed by modern deepfake videos make it increasingly difficult for detection methods. I evaluate that claim by testing the detection performance of four single CNN models as well as six stacked ensembles on three modern deepfakes datasets. I compare various ensemble approaches to combine single models and in what way their predictions should be incorporated into the ensemble output. The results I found was that the best approach for deepfakes detection is to create an ensemble, though, the ensemble approach plays a crucial role in the detection performance. The final proposed solution is an ensemble of all available single models which use the concept of soft (weighted) voting to combine its base-learners’ predictions. Results show that this proposed solution significantly improved deepfakes detection performance and substantially outperformed all single models.


## 1. Project structure

In the root folder, you will find the main files used during the experiment. Those are `train.py` (for training single models), `test.py` (for evaluating single models), and `ensemble.py` (for creating and evaluating ensembles).

This project uses [Sacred](https://sacred.readthedocs.io/en/stable/experiment.html) for experiment management. Sacred will only be executed if the full experiment is initiated. 

- Folder: `./data/`

The **data** folder contains everything related to the datasets used in this experiment. 

After downloading the datasets, I moved all datasets to the root folder and ran the code in the file `./data/preprocessing/data_sorting.py`. This code moves all videos to the *videos* folder while creating a CSV file with the columns 'video_id','fake','original_dataset'.
However you choose to do this, the result should be a folder containing all videos and a CSV file with information about video ID and it's associated label (i.e. real/deepfake).

The remaining files in this folder are used for the pre-processing phase during the experiment.

- Folder: `./experiments/`

The **experiments** folder contains three shell scripts (Bash) that you can run on Unix systems. For Windows, you need to look at the code and run each of those files sequentially with the arguments presented. 

`run.sh` runs the full experiment (i.e. pre-processing, training single models, evaluating single models, and lastly creating and evaluating ensemble), `train_all.sh` trains all single models, `test_all.sh` evaluates all single models, `ensembles.sh` creates and evaluates all ensembles.

- Folder: `./models`

The **models** folder contains class code from other research projects to instantiate the single models used. You can read more about these projects below in section 4 about single models. 
The class codes are used to instantiate single models, then the pre-trained models are used, retrieved from each project's original authors. As some of these pre-trained models are of larger file sizes (too large to fit this repository), they need to be downloaded from links provided by the authors. See the text file `./models/pre-trained/readme.txt` for more information. 

After re-training the single models, they will be saved in the re-trained subfolder. The ensembles will use these re-trained models for ensembling. 

- Folder: `./results`

## 2. Setup

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


## 3. Datasets

1. Download the datasets.
2. Put all videos into the same directory as a CSV-file with information about each video (e.g. `./data/videos`). Alternately, take a look at the file `data_sorting.py` for the code used in this research to organise the data.

### Celeb-DF
The dataset can be downloaded [here](https://github.com/danmohaha/celeb-deepfakeforensics#download).

### DeepFakeDetection
The dataset can be downloaded together with the regular FaceForensics++ dataset found [here](https://github.com/ondyari/FaceForensics/#access).

### Deepfake Detection Challenge
The small sample training set was used during this experiment. There's a much larger full training set also available that can be used to replace this smaller sample set.
Both the small sample training set and the full training set can be downloaded [here](https://www.kaggle.com/c/deepfake-detection-challenge/data).


## 4. Single models

Here you can find the original

### (1) Capsule

- Reproduced from:
https://github.com/tonylins/pytorch-mobilenet-v2

- Original License: Apache License 2.0.

- Reference

### (2) DSP-FWA

### (3) Ictu Oculi

### (4) XceptionNet


## 5. Ensembles

...


## 6. Authors

Google Scholar Profile(s):

- [Angelica Gardner](https://scholar.google.com/citations?user=mwcuZfkAAAAJ)


## 7. Citation

This research was carried out while the author studied at Linneaus University, Sweden.

If you use this code or research as a reference, please cite:
```
...
```
or reference (IEEE):
```
...
```