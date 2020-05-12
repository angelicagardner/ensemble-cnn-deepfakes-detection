import os, csv, cv2
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
    results_path = None # path to output folder, will contain evaluation results
    models_path = None # path to the saved models
    train_csv = None  # path to train set CSV
    val_csv = None  # path to validation set CSV
    test_csv = None  # path to test set CSV
    epochs = 100  # number of times a model will go through the complete training set
    batch_size = 32 # the amount of data examples included in each epoch
    early_stopping = 10 # training is stopped early if the validation loss has not decrease further after this number of epochs
    model_name = None  # CNN model
    split_id = 1 # split id (int)

# Classes and functions
class Average(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Function used for training set
def train(model, dataloader, device, criterion, optimizer, batches_per_epoch=None):
    losses = Average()
    accuracies = Average()
    scores_array = []
    all_preds = []
    all_labels = []
    tqdm_loader = tqdm(dataloader)
    model.train()

    for i, data in enumerate(tqdm_loader):
        (inputs, labels), name = data
        current_video = name[0].rsplit('_frame', 1)[0]
        if i == 0:
            last_video = current_video
            print("\n\nTraining on video {}".format(last_video))
        else:
            if not current_video == last_video:
                print("\n\nTraining on video {}".format(last_video))
                last_video = current_video

        inputs = inputs.to(device)
        labels = labels.to(device)

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
        all_preds += list(F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy())
        all_labels += (list(labels.cpu().data.numpy()))
        tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)

    return {'labels': all_labels, 'preds': all_preds, 'losses': losses, 'acc': accuracies}

# Function used for validation or test sets
def test(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = Average()
    predictions = pd.DataFrame(columns=['video_id', 'label', 'prediction', 'score', 'loss'])

    for i, data in enumerate(tqdm(dataloader)):
        (inputs, labels), name = data
        current_video = name[0].rsplit('_frame', 1)[0]
        if i == 0:
            last_video = current_video
            labels_array = []
            scores_array = []

        labels_array.append(labels.data[0].item())
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            scores = F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy()
            scores_array.append(scores.mean())
            loss = criterion(outputs, labels)

        losses.update(loss.item(), inputs.size(0))

        if not current_video == last_video and i > 0:
            print("\nSaving predictions for video {}\n".format(last_video))

            scores_mean = 0.0
            scores_array_int = []
            for score in scores_array:
                scores_mean += score
                scores_array_int.append(int(round(score, 0)))
            scores_mean = scores_mean / len(scores_array)

            predictions = predictions.append(
                {'video_id': last_video,
                'label': labels.data[0].item(),
                'prediction': int(round(scores.mean(), 0)),
                'score': scores_mean,
                'loss': losses.avg}, 
                ignore_index=True)

            last_video = current_video

    return predictions

# Main function
@ex.automain
def main(data_path, splits_path, results_path, models_path, train_csv, val_csv, test_csv, epochs, batch_size, early_stopping, model_name, split_id, _run):

    # Constants
    SCORES_DIR = os.path.join(results_path, 'scores')
    BEST_MODEL_PATH = os.path.join(models_path, model_name + '.pth')
    RESULTS_CSV_PATH = os.path.join(results_path, 'results.csv')
    EXP_ID = _run._id

    # Create folder for saving scoring metrics
    if not os.path.exists(results_path + 'scores'):
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
        model.last_linear = nn.Linear(model.last_linear.in_features, 2)
        print("Model: XceptionNet")
        optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=[0.5,0.999])
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
    dataset_train = CSVDataset(data_path, splits_path + train_csv, 'image_id', 'deepfake', transform=transform)
    dataset_val = CSVDataset(data_path, splits_path + val_csv, 'image_id', 'deepfake', transform=transform)
    dataset_test = CSVDataset(data_path, splits_path + test_csv, 'image_id', 'deepfake', transform=transform)
    dataloader_train = DataLoader(dataset_train)
    dataloader_val = DataLoader(dataset_val)
    dataloader_test = DataLoader(dataset_test)

    # Training settings
    criterion = nn.CrossEntropyLoss()

    # Train model
    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print('\nTraining epoch {}/{} for model {}\n'.format(epoch + 1, epochs, model_name.upper()))
        epoch_train_results = train(model, dataloader_train, device, criterion, optimizer)

        auc = roc_auc_score(epoch_train_results['labels'], epoch_train_results['preds'])

        print("\nTraining loss: " + str(epoch_train_results['losses'].avg) + "\nTraining AUC: " + str(auc) + "\nTraining accuracy: " + str(int(epoch_train_results['acc'].avg * 100)) + '%\n')
        
        # Test model on validation set
        epoch_val_results = test(model, dataloader_val, device)

        predictions = []
        labels = []
        losses_val = Average()
        for index, row in epoch_val_results.iterrows():
            labels.append(row['label'])
            predictions.append(row['prediction'])
            losses_val.update(row['loss'])

        acc_val = accuracy_score(labels, predictions)
        auc_val = roc_auc_score(labels, predictions, labels=[0,1])

        print("\nValidation loss: " + str(losses_val.avg) + "\nValidation AUC: " + str(auc_val) + "\nValidation accuracy: " + str(int(acc_val * 100)) + '%\n')
        print('-' * 40 + '\n')

        if 'scheduler' in locals():
            scheduler.step(losses_val.avg) # SGD optimiser only

        # If validation results have improved, save model
        if auc > best_val_auc:
            best_val_auc = auc
            best_val_results = epoch_val_results
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model, BEST_MODEL_PATH)
            #torch.save(model.state_dict(), os.path.join(os.getcwd() + '/models/', 'mantra-net' + '.pth'))
        else:
            epochs_without_improvement += 1

        # If validation hasn't improved for x epochs, apply early stopping
        if epochs_without_improvement > early_stopping:
            last_val_result = epoch_val_results
            break

    # Test model by making predictions on test set
    test_results = test(model, dataloader_test, device)

    predictions = []
    labels = []
    for index, row in test_results.iterrows():
        labels.append(row['label'])
        predictions.append(row['prediction'])

    # Calculate evaluation metrics 
    acc_test = accuracy_score(labels, predictions)
    fpr, tpr, _ = roc_curve(labels, predictions)
    auc_test = roc_auc_score(labels, predictions, labels=[0,1])
    conf_matrix = confusion_matrix(labels, predictions, labels=[0,1])
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)

    plt.plot(fpr, tpr, label="Model: " + model_name.upper() + ", AUC="+str(round(auc_test, 2)))
    plt.legend(loc=4)
    plt.show()

    df_cm = pd.DataFrame(conf_matrix, range(2), range(2))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True)
    plt.show()

    # Save predictions
    print("\nSaving evaluation metrics for model {}\n".format(model_name.upper()))
    scores = pd.DataFrame(columns=['run_id', 'split_id', 'model', 'best_epoch', 'val_loss', 'val_acc', 'val_auc', 'test_acc', 'test_auc', 'test_spec', 'test_sens'])
    scores = scores.append(
                    {'run_id': EXP_ID,
                    'split_id': split_id,
                    'model': model_name,
                    'best_epoch': best_epoch,
                    'val_loss': losses_val.avg,
                    'val_acc': int(acc_val * 100),
                    'val_auc': auc_val,
                    'test_acc': int(acc_test * 100),
                    'test_auc': auc_test,
                    'test_spec': specificity,
                    'test_sens': sensitivity}, 
                    ignore_index=True)
    scores.to_csv(os.path.join(SCORES_DIR, 'scores.csv'), index=False, mode='a')