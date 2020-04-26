import os, csv, cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import pretrainedmodels as ptm
import dlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.dataset_loader import CSVDataset, CSVDatasetWithName

import joblib

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
    elif model_name == 'xception':
        model = ptm.xception(num_classes=1000, pretrained='imagenet')
        print("Model: Xception")
    elif model_name == 'vgg16':
        model = ptm.vgg16(num_classes=1000, pretrained='imagenet')
        print("Model: VGG16")
    elif model_name == 'vgg19':
        model = ptm.vgg19(num_classes=1000, pretrained='imagenet')
        print("Model: VGG19")
    elif model_name == 'resnet50':
        model = ptm.resnet50(num_classes=1000, pretrained='imagenet')
        print("Model: Resnet50")
    model.last_linear = nn.Linear(model.last_linear.in_features, 2)
    size = model.input_size[1]
    mean = model.mean
    std = model.std
    model.eval()
    model.to(device)    

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load dataset
    dataset = CSVDatasetWithName(data_path, splits + train_csv, 'image_id', 'deepfake', transform=transform)
    dataloader = DataLoader(dataset)

    # Go through dataset and make predictions
    predictions = pd.DataFrame(columns=['video', 'label', 'score'])
    for i, data in enumerate(tqdm(dataloader)):
        (inputs, labels), name = data
       
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            scores = F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy()
            
        predictions = predictions.append(
            {'video': name[0],
            'label': labels.data[0].item(),
            'score': scores.mean()}, 
            ignore_index=True)
        
        labels_array = predictions['label'].values.astype(int)
        scores_array = predictions['score'].values.astype(float)
        acc = accuracy_score(labels_array, np.where(scores_array >= 0.5, 1, 0))
        conf_matrix = confusion_matrix(labels_array, scores_array >= 0.5, labels=[0,1])
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)

        predictions.to_csv(path + '/predictions.csv', index=False)

    # Save sample image set