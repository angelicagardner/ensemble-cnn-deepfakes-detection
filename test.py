import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.Capsule import Capsule
from models.DSP_FWA import DSP_FWA
from models.Ictu_Oculi import Ictu_Oculi
from models.XceptionNet import Xception

from data.average import Average
from data.dataset_loader import CSVDataset

def test(model, data_path, splits_path, test_csv, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((model.input_size[1], model.input_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(model.mean, model.std)
    ])

    dataset = CSVDataset(data_path, splits_path + test_csv, 'frame_id', 'deepfake', transform=transform)
    dataloader = DataLoader(dataset)

    model.eval()
    model.to(device)
    criterion = nn.BCELoss()
    
    losses = Average()
    predictions = pd.DataFrame(columns=['frame_id', 'label', 'score'])

    for i, data in enumerate(tqdm(dataloader)):
        (inputs, labels), name = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            score = F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy()
            loss = criterion(outputs, labels)

        losses.update(loss.item(), inputs.size(0))

        predictions = predictions.append(
            {'frame_id': name[0],
            'label': labels.data[0].item(),
            'score': score.mean()},
            ignore_index=True)

    all_labels = predictions['label'].values.astype(int)
    all_scores = np.rint(predictions['score'].values).astype(int)

    # Calculate evaluation metrics
    auc = roc_auc_score(all_labels, all_scores, labels=[0,1])
    acc = accuracy_score(all_labels, all_scores)
    conf_matrix = confusion_matrix(all_labels, all_scores, labels=[0,1])
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
        
    return ({'loss': losses.avg, 'auc': auc, 'acc': acc, 'spec': specificity, 'sens': sensitivity, 'cm': conf_matrix}, predictions)

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of model to be tested', required=True, default=None)
    parser.add_argument('--models_path', help='Path to re-trained model', required=True, default=os.path.join(os.getcwd(), 'models/'))
    parser.add_argument('--models_retrained_path', help='path to load re-trained models', required=True, default=os.path.join(os.getcwd(), 'models/re_trained/'))
    parser.add_argument('--data_path', help='Path to folder where images/video frames are stored', required=True, default=os.path.join(os.getcwd(), 'data/images/'))
    parser.add_argument('--splits_path', help='Path to CSV files with information about train, validation, and test sets', required=True, default=os.path.join(os.getcwd(), 'data/splits/'))
    parser.add_argument('--test_csv', help="Test split CSV file", required=True, default=os.path.join(os.getcwd(), 'test.csv'))
    parser.add_argument('--output_path', help="Folder where the output results will be saved", default=os.path.join(os.getcwd(), 'results/'))

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model_name == 'capsule':
        model = Capsule()
    elif args.model_name == 'dsp-fwa':
        model = DSP_FWA()
    elif args.model_name == 'ictu_oculi':
        model = Ictu_Oculi()
    elif args.model_name == 'xceptionnet':
        model = Xception()
    model.load_state_dict(torch.load(os.path.join(args.models_retrained_path, args.model_name.upper() + './pth')))

    # Make predictions on test set
    test_results = test(model, args.data_path, args.splits_path, args.test_csv, device)

    # Save evaluation metrics
    print(test_results[0])
    test_results[1].to_csv(os.path.join(args.output_path + 'model_metrics/test/', 'scores_' + args.model_name + '_' + '.csv'), index=False)

if __name__ == '__main__':
    main()