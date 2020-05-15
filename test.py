import argparse, os, datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, average_precision_score
import pretrainedmodels as ptm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.average import Average
from data.dataset_loader import CSVDataset

def test(model, images_path, csv_path, csv_dataset, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((model.input_size[1], model.input_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(model.mean, model.std)
    ])

    dataset = CSVDataset(images_path, csv_path + csv_dataset, 'frame_id', 'deepfake', transform=transform)
    dataloader = DataLoader(dataset)

    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
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
    parser.add_argument('--model_name', type=str, help='Name of model to be tested', required=True)
    parser.add_argument('--model_path', type=str, help='Path to trained model', required=True)
    parser.add_argument('--images_path', type=str, help='Path to folder where images/video frames are stored', required=True)
    parser.add_argument('--csv_path', type=str, help='Path to folder which contains test split csv file', required=True)
    parser.add_argument('--csv_file', type=str, help="Path to test split csv file", required=True)
    parser.add_argument('--output_path', type=str, help="Folder where the output results will be saved", default='')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model_name == 'capsule':
        model =ptm.vgg19(num_classes=1000, pretrained='imagenet')
    elif args.model_name == 'dsp-fwa':
        model = ptm.resnet50(num_classes=1000, pretrained='imagenet')
    elif args.model_name == 'ictu-oculi':
        model = ptm.vgg16(num_classes=1000, pretrained='imagenet')
    elif args.model_name == 'mantra-net':
        model = ptm.vgg16(num_classes=1000, pretrained='imagenet')
    elif args.model_name == 'xceptionnet':
        model = ptm.xception(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, 2)
    model.load_state_dict(torch.load(os.getcwd() + args.model_path + args.model_name + '.pth'))

    # Make predictions on test set
    test_results = test(model, os.getcwd() + args.images_path, os.getcwd() + args.csv_path, args.csv_file, device)

    # Save evaluation metrics
    print(test_results[0])
    test_results[1].to_csv(os.path.join(os.getcwd() + args.output_path, 'scores_' + args.model_name + '_' + str(datetime.datetime.now().timestamp()) + '.csv'), index=False)

if __name__ == '__main__':
    main()