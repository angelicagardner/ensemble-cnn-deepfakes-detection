import os, csv, cv2, datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, average_precision_score
import pretrainedmodels as ptm
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.average import Average
from data.dataset_loader import CSVDataset

# Set up experiment
ex = Experiment()
fs = FileStorageObserver.create('results/experiments') # Create the results output folder
ex.observers.append(fs)

# Add default configurations
@ex.config
def cfg():
    data_path = None  # path to video frames (folder containing images)
    splits_path = None # path to CSV files with information about train, validation, and test splits
    output_path = None # path to output folder, will contain evaluation results
    models_path = None # path to the saved models
    train_csv = None  # path to train set CSV
    val_csv = None  # path to validation set CSV
    epochs = 100  # number of times a model will go through the complete training set
    batch_size = 32 # the amount of data examples included in each epoch
    early_stopping = 10 # training is stopped early if the validation loss has not decrease further after this number of epochs
    model_name = None  # CNN model
    split_id = 1 # split id (int)

# Function used for training set
def train(model, dataloader, device, criterion, optimizer=None, batches_per_epoch=None):
    tqdm_loader = tqdm(dataloader)

    losses = Average()
    accuracies = Average()
    all_labels = []
    all_scores = []
    predictions = pd.DataFrame(columns=['frame_id', 'label', 'score'])

    if not optimizer == None:
        model.train()
    else:
        model.eval()

    for i, data in enumerate(tqdm_loader):
        (inputs, labels), name = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        if not optimizer == None:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            losses.update(loss.item(), inputs.size(0))
            acc = torch.sum(preds == labels.data).item() / preds.shape[0]
            accuracies.update(acc)
            score = F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy()
            all_labels += (list(labels.cpu().data.numpy()))
            all_scores += list(score)
            tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)
        else:
            with torch.no_grad():
                outputs = model(inputs)
                score = F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy()
                all_labels += list(labels.cpu().data.numpy())
                all_scores += list(score)
                loss = criterion(outputs, labels)

            losses.update(loss.item(), inputs.size(0))
        
        predictions = predictions.append(
                {'frame_id': name[0],
                'label': labels.data[0].item(),
                'score': score.mean()}, 
                ignore_index=True)

    all_labels = np.rint(all_labels).astype(int)
    all_scores = np.rint(all_scores).astype(int)

    auc = roc_auc_score(all_labels, all_scores, labels=[0,1])
    if not optimizer == None:
        acc = accuracies.avg
    else:
        acc = accuracy_score(all_labels, all_scores)

    return ({'loss': losses.avg, 'auc': auc, 'acc': acc}, predictions)

# Main function
@ex.automain
def main(data_path, splits_path, results_path, models_path, train_csv, val_csv, epochs, batch_size, early_stopping, model_name, split_id, _run):

    # Constants
    SCORES_DIR = os.path.join(results_path, 'scores')
    BEST_MODEL_PATH = os.path.join(models_path, model_name + '.pth')
    EXP_ID = _run._id

    # Create folder for saving scoring metrics
    if not os.path.exists(results_path + 'model_metrics/train/'):
        os.makedirs(SCORES_DIR)

    # Disable threading to run functions sequentially
    cv2.setNumThreads(0)
    # CPU or GPU utilisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and initialize optimiser algorithm with model settings
    if model_name == 'capsule':
        model = ptm.vgg19(num_classes=1000, pretrained='imagenet')
        print("Model: Capsule")
        optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=[0.9,0.999])
    elif model_name == 'dsp-fwa':
        model = ptm.resnet50(num_classes=1000, pretrained='imagenet')
        print("Model: DSP-FWA")
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, min_lr=1e-5)
    elif model_name == 'ictu_oculi':
        model = ptm.vgg16(num_classes=1000, pretrained='imagenet')
        print("Model: Ictu Oculi")
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, min_lr=1e-5)
    elif model_name == 'mantranet':
        model = ptm.vgg16(num_classes=1000, pretrained='imagenet')
        print("Model: ManTra-Net")
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=[0.5,0.5])
    elif model_name == 'xceptionnet':
        model = ptm.xception(num_classes=1000, pretrained='imagenet')
        print("Model: XceptionNet")
        optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=[0.5,0.999])
    model.last_linear = nn.Linear(model.last_linear.in_features, 2)
    size = model.input_size[1]
    mean = model.mean
    std = model.std
    model.to(device)    

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load datasets
    dataset_train = CSVDataset(data_path, splits_path + train_csv, 'frame_id', 'deepfake', transform=transform)
    dataset_val = CSVDataset(data_path, splits_path + val_csv, 'frame_id', 'deepfake', transform=transform)
    dataloader_train = DataLoader(dataset_train)
    dataloader_val = DataLoader(dataset_val)

    # Training settings
    criterion = nn.CrossEntropyLoss()

    # Train model
    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print('\nTraining epoch {}/{} for model {}\n'.format(epoch + 1, epochs, model_name.upper()))
        
        # Train model on training set
        epoch_train_results = train(model, dataloader_train, device, criterion, optimizer)
        print("\nTraining log loss: " + str(epoch_train_results[0]['loss']) + "\nTraining AUC: " + str(epoch_train_results[0]['auc']) + "\nTraining accuracy: " + str(epoch_train_results[0]['acc'] * 100) + '%\n')
        
        # Test model on validation set
        epoch_val_results = train(model, dataloader_val, device, criterion)
        print("\nValidation loss: " + str(epoch_val_results[0]['loss']) + "\nValidation AUC: " + str(epoch_val_results[0]['auc']) + "\nValidation accuracy: " + str(epoch_val_results[0]['acc'] * 100) + '%\n')
        print('-' * 40 + '\n')

        if 'scheduler' in locals():
            scheduler.step(epoch_val_results[0]['loss']) # SGD optimiser only

        # If validation results have improved, save model and predictions
        if epoch_val_results[0]['auc'] > best_val_auc:
            best_val_auc = epoch_val_results[0]['auc']
            best_val_results = epoch_val_results
            best_epoch = epoch
            epochs_without_improvement = 0
            best_train_result = epoch_train_results[0]
            best_val_result = epoch_val_results[0]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            epoch_train_results[1].to_csv(os.path.join(results_path, 'model_predictions/train/train_predictions_' + model_name + '.csv'), index=False)
            epoch_val_results[1].to_csv(os.path.join(results_path + 'model_predictions/train/val_predictions_' + model_name + '.csv'), index=False)
        else:
            epochs_without_improvement += 1

        # If validation hasn't improved for x epochs, apply early stopping
        if epochs_without_improvement > early_stopping:
            break

    # Save scores
    print("\nSaving training and validation metrics for model {}\n".format(model_name.upper()))
    scores = pd.DataFrame(columns=['run_id', 'split_id', 'model', 'best_epoch', 'train_loss', 'train_acc', 'train_auc', 'val_loss', 'val_acc', 'val_auc'])
    scores = scores.append(
                    {'run_id': EXP_ID,
                    'split_id': split_id,
                    'model': model_name,
                    'best_epoch': best_epoch + 1,
                    'train_loss': best_train_result['loss'],
                    'train_acc': best_train_result['acc'],
                    'train_auc': best_train_result['auc'],
                    'val_loss': best_val_result['loss'],
                    'val_acc': best_val_result['acc'],
                    'val_auc': best_val_result['auc']},
                    ignore_index=True)
    scores.to_csv(os.path.join(SCORES_DIR, 'train_scores_' + model_name + '.csv'), index=False, header=False)