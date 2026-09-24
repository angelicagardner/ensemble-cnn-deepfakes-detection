# Stronger Together? An Ensemble of CNNs for Deepfakes Detection.

This project contains the source code of the experiment described in *'Stronger Together? An Ensemble of CNNs for Deepfakes Detection'* which was my degree project at bachelor level at Linnaeus University during spring semester 2020. As the class has ended, this project is not active and the repository is archived.

In 2026, the source code was reconstructed so that it follows the experiment as it is described in the report (`docs/stronger_together.pdf`), since the code that was originally published was incomplete and could not be run. The reconstruction uses the same Python version and package versions as in 2020 and the single models are implemented from the original authors' source code and pre-trained models, as described in the report. However, this is not guaranteed to be the exact code that produced the results in the report and running it will not give identical results.

## Abstract

> Deepfakes technology is a face swap technique that enables anyone to replace faces in a video, with highly realistic results. Despite its usefulness, if used maliciously, this technique can have a significant impact on society, for instance, through the spreading of fake news or cyberbullying. This makes the ability of deepfakes detection a problem of utmost importance. In this paper, I tackle the problem of deepfakes detection by identifying deepfakes forgeries in video sequences. Inspired by the state-of-the-art, I study the ensembling of different machine learning solutions built on convolutional neural networks (CNNs) and use these models as objects for comparison between ensemble and single model performances. Existing work in the research field of deepfakes detection suggests that escalated challenges posed by modern deepfake videos make it increasingly difficult for detection methods. I evaluate that claim by testing the detection performance of four single CNN models as well as six stacked ensembles on three modern deepfakes datasets. I compare various ensemble approaches to combine single models and in what way their predictions should be incorporated into the ensemble output. The results I found was that the best approach for deepfakes detection is to create an ensemble, though, the ensemble approach plays a crucial role in the detection performance. The final proposed solution is an ensemble of all available single models which use the concept of soft (weighted) voting to combine its base-learners’ predictions. Results show that this proposed solution significantly improved deepfakes detection performance and substantially outperformed all single models.


## 1. Project structure

In the root folder, you will find the main files used during the experiment. Those are `train.py` (for training single models), `test.py` (for evaluating single models), and `ensemble.py` (for creating and evaluating ensembles). The file `metrics.py` contains the evaluation used by all three: predictions per video frame, and averaging the frame scores into a prediction per video.

This project uses [Sacred](https://sacred.readthedocs.io/en/stable/experiment.html) for experiment management. Sacred will only be executed if the full experiment is initiated.

- :file_folder: `./data/`

The **data** folder contains everything related to the datasets used in this experiment. `./data/preprocessing/data_sorting.py` moves all videos to the *videos* folder while creating a CSV file with the columns 'video_id','fake','original_dataset' (see section 3 below). `./data/split.py` splits the videos into train, validation, and test subsets (8:1:1), separates each video into frames (one frame every 0.5 seconds), and saves the face area of each frame as an image in `./data/images/`. The splits are saved as CSV files in `./data/splits/`.

- :file_folder: `./models/`

The **models** folder contains class code from other research projects to instantiate the single models used. You can read more about these projects below in section 4 about single models. Each file ends with a wrapper class used in this experiment, which gives the model a binary output (real/deepfake) and loads the pre-trained model from the original authors.
The pre-trained models need to be downloaded from links provided by the authors, see the text file `./models/pre_trained/readme.txt`.

After re-training the single models, they will be saved in the `./models/re_trained/` subfolder. The ensembles will use these re-trained models.

- :file_folder: `./results/`

The **results** folder contains the outputs from the experiment.

For single models, `./results/model_metrics/` contains evaluation metric values (training: per epoch and for the saved model; test: accuracy, confusion matrix, sensitivity, and specificity per video, AUC per video frame, and the ROC curve) and `./results/model_predictions/` contains the predictions for every video frame.

After evaluating the ensembles, their evaluation metrics, ROC curves, predictions, and the weights given to each base-learner (soft voting) will be saved in the `./results/ensemble/` subfolder.

- :file_folder: `./scripts/`

The **scripts** folder contains three shell scripts (Bash) that you can run on Unix systems from the root folder, e.g. `bash scripts/run.sh`. For Windows, run the Python files sequentially with the arguments presented in the scripts: `data/split.py`, `train.py`, `test.py`, `ensemble.py`.

`run.sh` runs the full experiment (i.e. pre-processing, training single models, evaluating single models, and lastly creating and evaluating ensembles), `train_all.sh` trains all single models, and `test_all.sh` evaluates all single models.

## 2. Setup

The experiment was run with Python 3.7.7. Run `pip install -r requirements.txt` (make sure pip represents Python 3.7).

Note: dlib 19.19.0 is built from source during installation and requires CMake and a C++ compiler. With newer compilers (e.g. GCC 13), the build fails on a missing include in dlib's bundled pybind11. This can be solved by adding `#include <cstdint>` as the first line of `dlib/external/pybind11/include/pybind11/attr.h` in the downloaded dlib source and installing it with `python setup.py install`.

### Configurations

Each shell script in `./scripts/` contains the arguments provided to the Python files. These arguments represent configurations and settings and can be left out if the default values should be used. The files are run with Sacred, e.g. `python train.py with model_name=capsule epochs=25 batch_size=64`.

Settings and configurations used:

```
- data_path=<path to video frames (folder containing images)>

- splits_path=<path to CSV files with information about train, validation, and test splits>

- output_path=<path to output folder where the results should be stored>

- models_pretrained_path=<path to load pre-trained models>

- models_output_path / models_retrained_path / models_saved_path=<path to the re-trained models>

- train_csv / val_csv / test_csv=<train, validation, and test CSV file>

- epochs=<number of times a model will go through the complete training set>

- batch_size=<the amount of data examples included in each iteration>

- early_stopping=<training is stopped early if the validation loss has not decreased further after this number of epochs, 0 = not used>

- model_name=<single model name: capsule, dsp-fwa, ictu_oculi, or xceptionnet>
```

The optimisation settings of each single model (optimiser, learning rate, and learning rate decay) are found in `TRAINING_SETTINGS` in `train.py`.

## 3. Datasets

1. Download the datasets.
2. Place each dataset in the data folder with one sub-folder for real videos and one for deepfake videos: `./data/celeb-df/`, `./data/deepfakedetection/`, and `./data/deepfake-detection-challenge/`, each containing a `real` and a `fake` folder with the .mp4 videos.
3. Run `python data/preprocessing/data_sorting.py`. This puts all videos into `./data/videos/` together with the CSV file `videos.csv` containing information about each video.

### Celeb-DF
The dataset (v2) can be downloaded [here](https://github.com/danmohaha/celeb-deepfakeforensics#download).

### DeepFakeDetection
The dataset can be downloaded together with the regular FaceForensics++ dataset found [here](https://github.com/ondyari/FaceForensics/#access).

### Deepfake Detection Challenge
The small sample training set was used during this experiment. There's a much larger full training set also available that can be used to replace this smaller sample set. The labels for the videos are found in the file metadata.json.
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

- Reproduced (ported from TensorFlow to PyTorch) from:
https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi/blob/master/blink_net.py and https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi/blob/master/deep_base/vgg16.py

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

For ensemble building, the package DeepStack was used. You can find more information about it [here](https://github.com/jcborges/DeepStack). Hard (majority) voting uses the StackEnsemble class with a majority vote as meta-model, and soft (weighted) voting uses the DirichletEnsemble class, which finds the weights of its members on the validation set. All ensembles are evaluated on the test set.

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
