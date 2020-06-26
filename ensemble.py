import argparse, os, csv
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepstack.base import Member
from deepstack.ensemble import StackEnsemble, DirichletEnsemble
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.Capsule import Capsule
from models.DSP_FWA import DSP_FWA
from models.Ictu_Oculi import Ictu_Oculi
from models.XceptionNet import Xception

from data.dataset_loader import CSVDataset

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_saved_path', type=str, help='Path to saved re-trained single models', required=True, default=os.path.join(os.getcwd(), 'models/re_trained/'))
    parser.add_argument('--models_train_predictions_path', type=str, help='Path to finding the training and validation predictions from single models', required=True, default=os.path.join(os.getcwd(), 'results/model_predictions/train/'))
    parser.add_argument('--models_test_predictions_path', type=str, help='Path to finding the test predictions from single models', required=True, default=os.path.join(os.getcwd(), 'results/model_predictions/test/'))
    parser.add_argument('--output_path', type=str, help='Path to output folder', required=True)
    parser.add_argument('--data_path', type=str, help='Path to folder where images/video frames are stored', required=True)
    parser.add_argument('--splits_path', type=str, help='path to CSV files with information about train, validation, and test splits', required=True)
    parser.add_argument('--csv_file', type=str, help="Path to test split csv file", required=True)

    args = parser.parse_args()

    members = []

    # Load base-learners
    for model in os.listdir(args.models_saved_path):
        if model.endswith('.pth'):
            model_name = model.rsplit('.pth', 1)[0]
            if model_name == 'capsule':
                model = Capsule()
            elif model_name == 'dsp-fwa':
                model = DSP_FWA()
            elif model_name == 'ictu_oculi':
                model = Ictu_Oculi()
            elif model_name == 'xceptionnet':
                model = Xception()
            model.load_state_dict(torch.load(os.path.join(args.models_saved_path, model_name + '.pth')))

            # Load model training and validation results
            train_labels = []
            train_predictions = []
            val_labels = []
            val_predictions = []
            for file in os.listdir(args.models_train_predictions_path):
                if file.endswith('.csv') and model_name in file:
                    with open(os.path.join(args.models_train_predictions_path, file)) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        i = 0
                        for row in csv_reader: 
                            if not i == 0:
                                if file.startswith('train'):
                                    train_labels.append(int(row[1]))
                                    train_predictions.append(int(round(float(row[2]))))
                                elif file.startswith('val'):
                                    val_labels.append(int(row[1]))
                                    val_predictions.append(int(round(float(row[2]))))
                            i += 1
           
            # Get predictions on test data
            test_predictions = []
            for file in os.listdir(args.models_test_predictions_path):
                if file.endswith('.csv') and model_name in file:
                    with open(os.path.join(args.models_test_predictions_path, file)) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        i = 0
                        for row in csv_reader: 
                            if not i == 0:
                                if file.startswith('test'):
                                    test_predictions.append(int(round(float(row[2]))))
                            i += 1
            
            # Create an ensemble member from the model
            member = Member(name=model_name.capitalize(), train_probs=train_predictions, train_classes=train_labels, val_probs=val_predictions, val_classes=val_labels, submission_probs=test_predictions)
            members.append(member)
    
    # Create and evaluate ensembles
    ensemble_names = ['best_hard', 'best_soft', 'small_hard', 'small_soft', 'all_hard', 'all_soft']

    for ensemble in ensemble_names:
        # Initialize ensemble (hard/majority vs. soft/weighted voting)
        if 'hard' in ensemble:
            ensemble = StackEnsemble()
        else:
            ensemble = DirichletEnsemble()
        # Add ensemble members
        if 'best' in ensemble: 
            ensemble.add_member(members[0])
            ensemble.add_member(members[2])
        elif 'small' in ensemble:
            ensemble.add_member(members[1])
            ensemble.add_member(members[2])
        elif 'all' in ensemble:
            ensemble.add_members(members)
        # Train the ensemble
        ensemble.fit()
        # Save ensemble evaluations
        print(ensemble.describe())

if __name__ == '__main__':
    main()