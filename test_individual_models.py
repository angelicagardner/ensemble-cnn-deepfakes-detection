import os, csv

import cv2
import pandas as pd
import dlib
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.dataset_loader import CSVDataset, CSVDatasetWithName

from models.Capsule.Capsule import VggExtractor, CapsuleNet
from models.DSP_FWA.DSP_FWA import SPPNet
from models.XceptionNet.Xception import xception

# Set up experiment
ex = Experiment()
fs = FileStorageObserver.create('results') # Create the results output folder
ex.observers.append(fs)

# Add default configurations
@ex.config
def cfg():
    data_path = None  # path to video frames
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
def main(data_path, splits, train_csv, val_csv, test_csv, model_name, split_id, _run):

    path = os.getcwd()

    # Disable threading to run functions sequentially
    cv2.setNumThreads(0)
    # CPU or GPU utilisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_name == "capsule":
        vgg_ext = VggExtractor()
        model = CapsuleNet(2, 0)
        if torch.cuda.is_available():
            checkpoint = torch.load(path + '/models/Capsule/capsule_21.pt')
        else:
            checkpoint = torch.load(path + '/models/Capsule/capsule_21.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        print("Model: Capsule")
    elif model_name == 'dsp-fwa':
        model = SPPNet(backbone=50, num_class=2)
        if torch.cuda.is_available():
            checkpoint = torch.load(path + '/models/DSP_FWA/SPP-res50.pth')
        else:
            checkpoint = torch.load(path + '/models/DSP_FWA/SPP-res50.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['net'])
        print("Model: DSP-FWA")
    elif model_name == 'xceptionnet':
        model = xception()
        print("Model: Xception")
    model.eval()
    model.to(device)    

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CSVDatasetWithName(data_path, splits + train_csv, 'image_id', 'deepfake', transform=transform)
    dataloader = DataLoader(dataset)

    predictions = pd.DataFrame(columns=['image', 'label', 'score'])

    for i, data in enumerate(tqdm(dataloader)):
       (inputs, labels), name = data

    # Save sample image set