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
fs = FileStorageObserver.create('results') # Create the results output folder
ex.observers.append(fs)

# Add default configurations
@ex.config
def cfg():
    data_path = None  # path to video frames (folder containing images)
    splits_path = None # path to CSV files with information about train, validation, and test splits
    results_path = None # path to output folder, will contain evaluation results
    train_csv = None  # path to train set CSV
    val_csv = None  # path to validation set CSV
    test_csv = None  # path to test set CSV
    epochs = 100  # number of times a model will go through the complete training set
    early_stopping = 10 # training is stopped early if the validation loss has not decrease further after this number of epochs
    model_name = None  # CNN model

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
def train(model, dataloader, device, criterion, optimizer):
    losses = Average()
    accuracies = Average()
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
            all_preds += list(F.softmax(outputs, dim=1)[:,1].cpu().data.numpy())
            all_labels += list(labels.cpu().data.numpy())
            tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)

    return {'labels': all_labels, 'preds': all_preds, 'losses': losses, 'acc': accuracies}

# Function used for validation or test sets
def test(model, dataloader, device, results_path=None, model_name=None):
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
        else:
            if not current_video == last_video:
                print("\n\nSaving predictions for video {}".format(last_video))

                scores_mean = 0.0
                scores_array_int = []
                for score in scores_array:
                    scores_mean += score
                    scores_array_int.append(int(round(score, 0)))
                scores_mean = scores_mean / len(scores_array)

                predictions = predictions.append(
                    {'video_id': last_video,
                    'label': labels.data[0].item(),
                    'prediction': int(round(score, 0)),
                    'score': scores_mean,
                    'loss': losses.avg}, 
                    ignore_index=True)

                if not model_name == None:
                    predictions.to_csv(results_path + 'predictions/predictions_test_{}.csv'.format(model_name), index=False)
                else:
                    predictions.to_csv(results_path + 'predictions/predictions_val_{}.csv'.format(model_name), index=False)
                last_video = current_video

        labels_array.append(labels.data[0].item())
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            scores = F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy()
            scores_array.append(scores.mean())
            loss = criterion(outputs, labels)

        losses.update(loss.item(), inputs.size(0))

# Main function
@ex.automain
def main(data_path, splits_path, results_path, train_csv, val_csv, test_csv, epochs, early_stopping, model_name, _run):
    
    path = os.getcwd()
    models_path = path + '/models/' + model_name + '.pth'
    if not os.path.exists(results_path + 'predictions'):
        os.mkdir(results_path + 'predictions')
        os.mkdir(results_path + 'scores')

    # Disable threading to run functions sequentially
    cv2.setNumThreads(0)
    # CPU or GPU utilisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_name == 'xceptionnet':
        model = ptm.xception(num_classes=1000, pretrained='imagenet')
        print("Model: XceptionNet")
    elif model_name == 'vgg16':
        model = ptm.vgg16(num_classes=1000, pretrained='imagenet')
        print("Model: VGG16")
    elif model_name == 'vgg19':
        model = ptm.vgg19(num_classes=1000, pretrained='imagenet')
        print("Model: VGG19")
    elif model_name == 'dsp-fwa':
        model = ptm.resnet50(num_classes=1000, pretrained='imagenet')
        print("Model: DSP-FWA")
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
    dataset_train = CSVDataset(data_path, splits_path + train_csv, 'image_id', 'deepfake', transform=transform)
    dataset_val = CSVDataset(data_path, splits_path + val_csv, 'image_id', 'deepfake', transform=transform)
    dataset_test = CSVDataset(data_path, splits_path + test_csv, 'image_id', 'deepfake', transform=transform)
    dataloader_train = DataLoader(dataset_train)
    dataloader_val = DataLoader(dataset_val)
    dataloader_test = DataLoader(dataset_test)
    # TODO: Save sample image set

    # Training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-5, patience=8)

    # Train model
    metrics = {
        'train': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc']),
        'val': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc'])
    }

    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print('Training epoch {}/{} for model {}\n'.format(epoch + 1, epochs, model_name.upper()))
        epoch_train_results = train(model, dataloader_train, device, criterion, optimizer)

        auc = roc_auc_score(epoch_train_results['labels'], epoch_train_results['preds'])

        metrics['train'] = metrics['train'].append(
            {'epoch': epoch,
            'loss': epoch_train_results['losses'].avg,
            'acc':epoch_train_results['accuracies'].avg,
            'auc': auc}, 
            ignore_index=True)
        print("Training loss: " + str(epoch_train_results['losses'].avg) + "\nTraining AUC: " + str(auc) + "\nTraining accuracy: " + str(epoch_train_results['accuracies'].avg))

        # Testing model on validation set
        test(model, dataloader_train, device)

        preds = pd.read_csv(results_path + 'predictions/predictions_val_{}.csv'.format(model_name))
        predictions = []
        labels = []
        losses = Average()
        for i, row in enumerate(tqdm(preds.values)):
            video_id, label, prediction, score, loss = row
            labels.append(label)
            predictions.append(prediction)
            losses.append(loss)

        acc = accuracy_score(labels, predictions)
        auc = roc_auc_score(labels, predictions, labels=[0,1])

        metrics['val'] = metrics['val'].append(
            {'epoch': epoch,
            'acc': acc,
            'auc': auc}, 
            ignore_index=True)
        print("Validation loss: " + str(losses.avg) + "\Validation AUC: " + str(auc) + "\Validation accuracy: " + str(acc))
        print('-' * 40)

        scheduler.step(loss)

    # Save trained model
    torch.save(model, models_path)

    # Test model: Make predictions on test set
    test(model, dataloader_train, device, results_path, model_name)

    # Test model: Save evaluation metrics  
    preds = pd.read_csv(results_path + 'predictions/predictions_test_{}.csv'.format(model_name))
    predictions = []
    labels = []
    for i, row in enumerate(tqdm(preds.values)):
        video_id, label, prediction, score, loss = row
        labels.append(label)
        predictions.append(prediction)

    scores = pd.DataFrame(columns=['accuracy', 'auc', 'specificity', 'sensitivity'])
    acc = accuracy_score(labels, predictions)
    fpr, tpr, _ = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions, labels=[0,1])
    conf_matrix = confusion_matrix(labels, predictions, labels=[0,1])
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)

    plt.plot(fpr, tpr, label="Model: " + model_name.capitalize() + ", AUC="+str(round(auc, 2)))
    plt.legend(loc=4)
    plt.show()

    df_cm = pd.DataFrame(conf_matrix, range(2), range(2))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True)
    plt.show()

    print("\nSaving evaluation metrics for model {}\n".format(model_name.capitalize()))
    scores = scores.append(
                    {'accuracy': int(acc * 100),
                    'auc': round(auc, 2),
                    'specificity': specificity,
                    'sensitivity': sensitivity}, 
                    ignore_index=True)
    scores.to_csv(results_path + 'scores/scores_{}.csv'.format(model_name), index=False)