import os

import cv2
import pandas as pd
import pretrainedmodels as ptm
import torch
from torch.utils.data import DataLoader

from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.dataset_loader import CSVDataset

from models.XceptionNet.Xception import Xception

ex = Experiment()
fs = FileStorageObserver.create('results') # Creates the results output folder
ex.observers.append(fs)

# Add default configurations
@ex.config
def cfg():
    train_root = None  # path to train images
    train_csv = None  # path to train CSV
    val_root = None  # path to validation images
    val_csv = None  # path to validation CSV
    test_root = None  # path to test images
    test_csv = None  # path to test CSV
    epochs = 30  # number of epochs
    batch_size = 32  # batch size
    num_workers = 8  # parallel jobs for data loading and augmentation
    model_name = None  # model
    split_id = None  # split id (int)

# Training functions

# Main function
@ex.automain
def main(train_root, train_csv, val_root, val_csv, test_root, test_csv, model_name, split_id, _run):

    # Disable threading to run functions sequentially
    cv2.setNumThreads(0)
    # CPU or GPU utilisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate ANN model
    if model_name == 'xceptionnet':
        model = Xception()
    elif model_name == 'meso4':
        model = Xception()
    model.to(device)    