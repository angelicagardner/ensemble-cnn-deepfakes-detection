import os
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.dataset_loader import CSVDataset

# Ensemble class
class Ensemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(4, 2)

    def forward(self, x_test):
        x_modelA = self.modelA(x_test)
        x_modelB = self.modelB(x_test)
        x = torch.cat((x_modelA, x_modelB), dim=1)
        x = self.classifier(F.relu(x))
        return x

# Load base-learners
xception = torch.load(os.getcwd() + '/models/xceptionnet.pth')
size = xception.input_size[1]
xception.eval()

ensemble = Ensemble(xception, xception)

transform = transforms.Compose([
    transforms.ToTensor(),
])

data = CSVDataset(os.getcwd() + '/data/images/', os.getcwd() + '/data/splits/train.csv', 'image_id', 'deepfake', transform=transform)
dataloader = DataLoader(data)

all_labels = []
all_predictions = []
for i, data_example in enumerate(tqdm(dataloader)):
    (inputs, labels), name = data_example
    current_video = name[0].rsplit('_frame', 1)[0]
    label = labels.data[0].item()
    all_labels.append(label)
    output = ensemble(inputs)
    score = F.softmax(output, dim=1)[:, 1].cpu().data.numpy()
    pred = int(round(float(score)))
    all_predictions.append(pred)
    print(label == pred)

acc = accuracy_score(all_labels, all_predictions)
print(acc)