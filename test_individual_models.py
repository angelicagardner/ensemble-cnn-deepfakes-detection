import os, csv

import cv2
import pandas as pd
import dlib
import torch
from torch.utils.data import DataLoader

from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.dataset_loader import CSVDataset

from models.XceptionNet.Xception import xception

# Set up experiment
ex = Experiment()
fs = FileStorageObserver.create('results') # Create the results output folder
ex.observers.append(fs)

# Add default configurations
@ex.config
def cfg():
    videos = None  # path to videos
    splits = None # path to split information
    train_csv = None  # path to train CSV
    val_csv = None  # path to validation CSV
    test_csv = None  # path to test CSV
    epochs = 30  # number of epochs
    batch_size = 32  # batch size
    num_workers = 8  # parallel jobs for data loading and augmentation
    model_name = None  # model
    split_id = None  # split id (int)

# Main function
@ex.automain
def main(videos, splits, train_csv, val_csv, test_csv, model_name, split_id, _run):

    path = os.getcwd()

    # Disable threading to run functions sequentially
    cv2.setNumThreads(0)
    # CPU or GPU utilisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_name == 'xceptionnet':
        model = xception()
        print("Model: Xception")
    model.to(device)    

    # Go through dataset
    with open(path + splits + test_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                video_name = row[0]
                video = cv2.VideoCapture(path + videos + video_name)

                face_detector = dlib.get_frontal_face_detector()

            line_count += 1