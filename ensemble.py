import argparse, os, csv
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, average_precision_score
import pretrainedmodels as ptm
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepstack.base import Member
from deepstack.ensemble import StackEnsemble
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.dataset_loader import CSVDataset
from test import test

import numpy as np

def main():

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path', type=str, help='Path to trained models', required=True)
    parser.add_argument('--models_results_path', type=str, help='Path to finding the training and validation results for single models', required=True)
    parser.add_argument('--output_path', type=str, help='Path to output folder', required=True)
    parser.add_argument('--images_path', type=str, help='Path to folder where images/video frames are stored', required=True)
    parser.add_argument('--csv_path', type=str, help='Path to folder which contains test split csv file', required=True)
    parser.add_argument('--csv_file', type=str, help="Path to test split csv file", required=True)

    args = parser.parse_args()

    members = []

    # Load base-learners
    for model in os.listdir(os.getcwd() + args.models_path):
        if model.endswith('.pth'):
            model_name = model.rsplit('.pth', 1)[0]
            if model_name == 'capsule':
                model =ptm.vgg19(num_classes=1000, pretrained='imagenet')
            elif model_name == 'dsp-fwa':
                model = ptm.resnet50(num_classes=1000, pretrained='imagenet')
            elif model_name == 'ictu_oculi':
                model = ptm.vgg16(num_classes=1000, pretrained='imagenet')
            elif model_name == 'mantranet':
                model = ptm.vgg16(num_classes=1000, pretrained='imagenet')
            elif model_name == 'xceptionnet':
                model = ptm.xception(num_classes=1000, pretrained='imagenet')
                model.last_linear = nn.Linear(model.last_linear.in_features, 2)
            model.load_state_dict(torch.load(os.getcwd() + args.models_path + '/' + model_name + '.pth'))

            # Load training and validation results
            train_labels = []
            train_predictions = []
            val_labels = []
            val_predictions = []
            for file in os.listdir(os.getcwd() + args.models_results_path):
                if file.endswith('.csv'):
                    with open(os.path.join(os.getcwd() + args.models_results_path, file)) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        for row in csv_reader: 
                            if file.startswith('train'):
                                train_labels.append(row[1])
                                train_predictions.append(row[2])
                            elif file.startswith('val'):
                                val_labels.append(row[1])
                                val_predictions.append(row[2])

            # Get predictions on test data
            test_predictions = test(model, os.getcwd() + args.images_path, os.getcwd() + args.csv_path, args.csv_file)

            # Create an ensemble member from the model
            member = Member(name=model_name.capitalize(), train_probs=train_predictions, train_classes=train_labels, val_probs=val_predictions, val_classes=val_labels, submission_probs=test_predictions['predictions'])
            members.append(member)

            acc = accuracy_score(test_predictions['labels'], test_predictions['predictions'])
            fpr, tpr, _ = roc_curve(test_predictions['labels'], test_predictions['predictions'])
            auc = roc_auc_score(test_predictions['labels'], test_predictions['predictions'], labels=[0,1])

    # Initialise ensemble
    stack = StackEnsemble()
    stack.add_members(members)

    # Train the ensemble
    stack.fit()

    # Get evaluation metrics
    stack.describe()

    # Save ensemble
    stack.save(os.getcwd() + '/ensemble')

if __name__ == '__main__':
    main()