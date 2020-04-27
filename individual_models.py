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

from data.dataset_loader import CSVDataset, CSVDatasetWithName

# Set up experiment
ex = Experiment()
fs = FileStorageObserver.create('results') # Create the results output folder
ex.observers.append(fs)

# Add default configurations
@ex.config
def cfg():
    data_path = None  # path to video frames
    splits_path = None # path to split information
    results_path = None # path to output results
    train_csv = None  # path to train CSV
    val_csv = None  # path to validation CSV
    test_csv = None  # path to test CSV
    epochs = 30  # number of epochs
    batch_size = 32  # batch size
    num_workers = 8  # parallel jobs for data loading and augmentation
    model_name = None  # model

# Main function
@ex.automain
def main(data_path, splits_path, results_path, train_csv, val_csv, test_csv, epochs, model_name, _run):

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
    if model_name == 'xception':
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
        print("Model: ResNet50")
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
    dataset = CSVDatasetWithName(data_path, splits_path + train_csv, 'image_id', 'deepfake', transform=transform)
    dataloader = DataLoader(dataset)
    # TODO: Save sample image set

    # Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-5, patience=8)

    metrics = {
        'train': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc']),
        'val': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc'])
    }

    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print('Training epoch {}/{} for model {}'.format(epoch + 1, epochs, model_name.capitalize()))

    # Save trained model
    torch.save(model, models_path)

    # Test model: Make predictions on test set
    predictions = pd.DataFrame(columns=['video_id', 'label', 'prediction', 'score'])
    for i, data in enumerate(tqdm(dataloader)):
        (inputs, labels), name = data
        current_video = name[0].rsplit('_frame', 1)[0]
        if i == 0:
            last_video = current_video
            labels_array = []
            scores_array = []
        else:
            if not current_video == last_video:
                print("\nSaving predictions for video {}".format(last_video))
        
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
                    'score': scores_mean}, 
                    ignore_index=True)

                predictions.to_csv(results_path + 'predictions/predictions_{}.csv'.format(model_name), index=False)
                last_video = current_video

        labels_array.append(labels.data[0].item())

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            scores = F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy()
            scores_array.append(scores.mean())

    # Test model: Save evaluation metrics  
    preds = pd.read_csv(results_path + 'predictions/predictions_{}.csv'.format(model_name))
    predictions = []
    labels = []
    for i, row in enumerate(tqdm(preds.values)):
        video_id, label, prediction, score = row
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