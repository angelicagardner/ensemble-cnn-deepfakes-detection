# Stronger Together? An Ensemble of CNNs for Deepfakes Detection.

This project contains the source code of the experiment described in *'Stronger Together? An Ensemble of CNNs for Deepfakes Detection'* which was my degree project at bachelor level at Linnaeus University during spring semester 2020. As the class has ended, this project is currently not active and I consider this repository as under mantainance. However, do feel free to open an issue if you encounter problems or something that needs fixing.


## Abstract

> Deepfakes technology is a face swap technique that enables anyone to replace faces in a video, with highly realistic results. Despite its usefulness, if used maliciously, this technique can have a significant impact on society, for instance, through the spreading of fake news or cyberbullying. This makes the ability of deepfakes detection a problem of utmost importance. In this paper, I tackle the problem of deepfakes detection by identifying deepfakes forgeries in video sequences. Inspired by the state-of-the-art, I study the ensembling of different machine learning solutions built on convolutional neural networks (CNNs) and use these models as objects for comparison between ensemble and single model performances. Existing work in the research field of deepfakes detection suggests that escalated challenges posed by modern deepfake videos make it increasingly difficult for detection methods. I evaluate that claim by testing the detection performance of four single CNN models as well as six stacked ensembles on three modern deepfakes datasets. I compare various ensemble approaches to combine single models and in what way their predictions should be incorporated into the ensemble output. The results I found was that the best approach for deepfakes detection is to create an ensemble, though, the ensemble approach plays a crucial role in the detection performance. The final proposed solution is an ensemble of all available single models which use the concept of soft (weighted) voting to combine its base-learners’ predictions. Results show that this proposed solution significantly improved deepfakes detection performance and substantially outperformed all single models.


## 1. Project structure

In the root folder, you will find the main files used during the experiment. Those are `train.py` (for training single models), `test.py` (for evaluating single models), and `ensemble.py` (for creating and evaluating ensembles).

This project uses [Sacred](https://sacred.readthedocs.io/en/stable/experiment.html) for experiment management. Sacred will only be executed if the full experiment is initiated. 

- :file_folder: `./data/`

The **data** folder contains everything related to the datasets used in this experiment. 

After downloading the datasets, I moved all datasets to the root folder and ran the code in the file `./data/preprocessing/data_sorting.py`. This code moves all videos to the *videos* folder while creating a CSV file with the columns 'video_id','fake','original_dataset'.
However you choose to do this, the result should be a folder containing all videos and a CSV file with information about video ID and it's associated label (i.e. real/deepfake).

The remaining files in this folder are used for the pre-processing phase during the experiment.

- :file_folder: `./models/`

The **models** folder contains class code from other research projects to instantiate the single models used. You can read more about these projects below in section 4 about single models. 
The class codes are used to instantiate single models, then the pre-trained models are used, retrieved from each project's original authors. As some of these pre-trained models are of larger file sizes (too large to fit this repository), they need to be downloaded from links provided by the authors. See the text file `./models/pre-trained/readme.txt` for more information. 

After re-training the single models, they will be saved in the re-trained subfolder. The ensembles will use these re-trained models for ensembling. 

- :file_folder: `./results/`

The **results** folder contains the outputs from the experiment. All outputs from the Sacred experiment management is placed in `./results/experiments/` subfolder. 

For single models, subfolders `./results/model_metrics/` and `./results/models_predictions/` contain training and testing output data: single model evaluation metric values as well as video frame-level predictions respectively.

After evaluating the ensemble performances, those evaluation metrics will be saved in the `./results/ensemble/` subfolder.

- :file_folder: `./scripts/`

The **scripts** folder contains three shell scripts (Bash) that you can run on Unix systems. For Windows, you need to look at the code and run each of those files sequentially with the arguments presented. 

`run.sh` runs the full experiment (i.e. pre-processing, training single models, evaluating single models, and lastly creating and evaluating ensemble), `train_all.sh` trains all single models, `test_all.sh` evaluates all single models, `ensembles.sh` creates and evaluates all ensembles.

## 2. Setup

Run `pip install -r requirements.txt`
(Make sure pip represents python 3)


### Requirements

See file *requirements.txt* for necessary packages.

### Configurations

Each shell script in `./scripts/` contains arguments that should be provided to the files for execution. These arguments represent configurations and settings and can, in most cases, be left out if the program should use the default configurations (i.e. as the values used during this experiment).

Settings and configurations used:

```
- data_path=<path to video frames (folder containing images)>

- splits_path=<path to CSV files with information about train, validation, and test splits>

- output_path=<path to output folder where the results should be stored>

- models_path=<path to model classes>

- models_pretrained_path=<path to load pretrained models>

- train_csv=<train CSV file>

- val_csv=<validation CSV file>

- test_csv=<test CSV file>

- epochs=<number of times a model will go through the complete training set>

- batch_size=<the amount of data examples included in each iteration>

- early_stopping=<training is stopped early if the validation loss has not decrease further after this number of epochs>

- model_name=<single model name>
```


## 3. Datasets

1. Download the datasets.
2. Put all videos into the same directory as a CSV-file with information about each video (e.g. `./data/videos`). Alternately, take a look at the file `./data/preprocessing/data_sorting.py` for the code used in this research to organise the data.

### Celeb-DF
The dataset can be downloaded [here](https://github.com/danmohaha/celeb-deepfakeforensics#download).

### DeepFakeDetection
The dataset can be downloaded together with the regular FaceForensics++ dataset found [here](https://github.com/ondyari/FaceForensics/#access).

### Deepfake Detection Challenge
The small sample training set was used during this experiment. There's a much larger full training set also available that can be used to replace this smaller sample set.
Both the small sample training set and the full training set can be downloaded [here](https://www.kaggle.com/c/deepfake-detection-challenge/data).


## 4. Single models

The single models used in this experiment were reproduced from other research projects for deepfakes detection. Below is where you can find more information about those models.

### (1) Capsule

- Reproduced from:
https://github.com/nii-yamagishilab/Capsule-Forensics-v2/blob/master/model_big.py

- Original License: 
BSD 3-Clause License

- Reference:
H. H. Nguyen, J. Yamagishi, and I. Echizen, “Use of a Capsule Network to Detect Fake Images and Videos,” arXiv preprint arXiv:1910.12467. 2019 Oct 29.

### (2) DSP-FWA

- Reproduced from:
https://github.com/danmohaha/DSP-FWA/blob/master/py_utils/DL/pytorch_utils/models/classifier.py

- Original License:
https://github.com/danmohaha/DSP-FWA#notice

- Reference:
Li, Y., & Lyu, S. (2019). Exposing DeepFake Videos By Detecting Face Warping Artifacts. In IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).

### (3) Ictu Oculi

- Reproduced from:
https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi/blob/master/blink_net.py

- Original License:
https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi#notice

- Reference:
Li, Y., Chang, M.C., and Lyu, S. 2018. In Ictu Oculi: Exposing AI Generated Fake Face Videos by Detecting Eye Blinking. In IEEE International Workshop on Information Forensics and Security (WIFS).

### (4) XceptionNet

- Reproduced from:
https://github.com/ondyari/FaceForensics/blob/master/classification/network/xception.py

- Original License:
https://github.com/ondyari/FaceForensics/blob/master/LICENSE

- Reference:
Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nie\ssner 2019. FaceForensics++: Learning to Detect Manipulated Facial Images. In International Conference on Computer Vision (ICCV).


## 5. Ensembles

For ensemble building, the package DeepStack was used. You can find more information about it [here](https://github.com/jcborges/DeepStack). 

Six different ensembles are built during the experiment:

1. Two best performing single models, using hard voting.
2. Two best performing single models, using soft voting.
3. Two single models with the smallest file sizes, using hard voting.
4. Two single models with the smallest file sizes, using soft voting.
5. All single models, using hard voting.
6. All single models, using soft voting.


## 6. Authors

Google Scholar Profile(s):

- [Angelica Gardner](https://scholar.google.com/citations?user=mwcuZfkAAAAJ)


## 7. Citation

This bachelor's degree project was carried out while the author studied at [Linnaeus University the Faculty of Technology, Department of Computer Science](https://lnu.se/en/meet-linnaeus-university/Organisation/faculty-of-technology/) in Sweden.

If you use anything from this study as a reference, please cite:
```
@book{gardner_2020, 
journal={Stronger Together? An Ensemble of CNNs for Deepfakes Detection}, 
author={Gardner, Angelica}, 
year={2020}}
```
