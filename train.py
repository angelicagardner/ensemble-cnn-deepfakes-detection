import os

import cv2
import pandas as pd
import pretrainedmodels as ptm
import torch

from sacred import Experiment

from data.dataset_loader import CSVDataset

from models.XceptionNet.Xception import Xception

ex = Experiment()

# Add configurations
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

# Main function
@ex.automain
def main(train_root, val_root, test_root, model_name, _run):
    assert(model_name in ('xceptionnet'))

    cv2.setNumThreads(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'xceptionnet':
        model = Xception()
    model.to(device)

    datasets = {
        'train': CSVDataset(train_root),
        'val': CSVDataset(val_root),
        'test': CSVDataset(test_root)
    }

    dataloaders = {}