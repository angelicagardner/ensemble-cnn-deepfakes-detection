import os
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

# Load Capsule base-learner
capsule = ptm.vgg19(num_classes=1000, pretrained='imagenet')
capsule.load_state_dict(torch.load(os.getcwd() + '/models/capsule.pth'))
capsule_size = capsule.input_size[1]
mean = capsule.mean
std = capsule.std
capsule.eval()

transform = transforms.Compose([
    transforms.Resize((capsule_size, capsule_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

data = CSVDataset(os.getcwd() + '/data/images/', os.getcwd() + '/data/splits/test.csv', 'image_id', 'deepfake', transform=transform)
dataloader = DataLoader(data)

all_labels = []
all_predictions = []
for i, data_example in enumerate(tqdm(dataloader)):
    (inputs, labels), name = data_example
    current_video = name[0].rsplit('_frame', 1)[0]
    label = labels.data[0].item()
    all_labels.append(label)
    output = capsule(inputs)
    score = F.softmax(output, dim=1)[:, 1].cpu().data.numpy()
    pred = int(round(float(score)))
    all_predictions.append(pred)
    print(label == pred)

capsule_member = Member(name="Capsule", train_probs=all_predictions, train_classes=all_labels, val_probs=all_predictions, val_classes=all_labels, submission_probs=all_predictions)

# Load DSP-FWA base-learner
dsp_fwa = ptm.resnet50(num_classes=1000, pretrained='imagenet')
dsp_fwa.load_state_dict(torch.load(os.getcwd() + '/models/dsp-fwa.pth'))
dsp_fwa_size = dsp_fwa.input_size[1]
mean = dsp_fwa.mean
std = dsp_fwa.std
dsp_fwa.eval()

transform = transforms.Compose([
    transforms.Resize((dsp_fwa_size, dsp_fwa_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

data = CSVDataset(os.getcwd() + '/data/images/', os.getcwd() + '/data/splits/test.csv', 'image_id', 'deepfake', transform=transform)
dataloader = DataLoader(data)

all_labels = []
all_predictions = []
for i, data_example in enumerate(tqdm(dataloader)):
    (inputs, labels), name = data_example
    current_video = name[0].rsplit('_frame', 1)[0]
    label = labels.data[0].item()
    all_labels.append(label)
    output = dsp_fwa(inputs)
    score = F.softmax(output, dim=1)[:, 1].cpu().data.numpy()
    pred = int(round(float(score)))
    all_predictions.append(pred)
    print(label == pred)

dsp_fwa_member = Member(name="DSP-FWA", train_probs=all_predictions, train_classes=all_labels, val_probs=all_predictions, val_classes=all_labels, submission_probs=all_predictions)

# Load Ictu Oculi base-learner
ictu_oculi = ptm.vgg16(num_classes=1000, pretrained='imagenet')
ictu_oculi.load_state_dict(torch.load(os.getcwd() + '/models/ictu-oculi.pth'))
ictu_oculi_size = ictu_oculi.input_size[1]
mean = ictu_oculi.mean
std = ictu_oculi.std
ictu_oculi.eval()

transform = transforms.Compose([
    transforms.Resize((ictu_oculi_size, ictu_oculi_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

data = CSVDataset(os.getcwd() + '/data/images/', os.getcwd() + '/data/splits/test.csv', 'image_id', 'deepfake', transform=transform)
dataloader = DataLoader(data)

all_labels = []
all_predictions = []
for i, data_example in enumerate(tqdm(dataloader)):
    (inputs, labels), name = data_example
    current_video = name[0].rsplit('_frame', 1)[0]
    label = labels.data[0].item()
    all_labels.append(label)
    output = ictu_oculi(inputs)
    score = F.softmax(output, dim=1)[:, 1].cpu().data.numpy()
    pred = int(round(float(score)))
    all_predictions.append(pred)
    print(label == pred)

ictu_oculi_member = Member(name="Ictu Oculi", train_probs=all_predictions, train_classes=all_labels, val_probs=all_predictions, val_classes=all_labels, submission_probs=all_predictions)

# Load ManTra-Net base-learner
mantranet = ptm.vgg16(num_classes=1000, pretrained='imagenet')
mantranet.load_state_dict(torch.load(os.getcwd() + '/models/mantra-net.pth'))
mantranet_size = mantranet.input_size[1]
mean = mantranet.mean
std = mantranet.std
mantranet.eval()

transform = transforms.Compose([
    transforms.Resize((mantranet_size, mantranet_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

data = CSVDataset(os.getcwd() + '/data/images/', os.getcwd() + '/data/splits/test.csv', 'image_id', 'deepfake', transform=transform)
dataloader = DataLoader(data)

all_labels = []
all_predictions = []
for i, data_example in enumerate(tqdm(dataloader)):
    (inputs, labels), name = data_example
    current_video = name[0].rsplit('_frame', 1)[0]
    label = labels.data[0].item()
    all_labels.append(label)
    output = mantranet(inputs)
    score = F.softmax(output, dim=1)[:, 1].cpu().data.numpy()
    pred = int(round(float(score)))
    all_predictions.append(pred)
    print(label == pred)

mantranet_member = Member(name="ManTra-Net", train_probs=all_predictions, train_classes=all_labels, val_probs=all_predictions, val_classes=all_labels, submission_probs=all_predictions)

# Load XceptionNet base-learner
xceptionnet = torch.load(os.getcwd() + '/models/xceptionnet.pth')
xceptionnet_size = xceptionnet.input_size[1]
mean = xceptionnet.mean
std = xceptionnet.std
xceptionnet.eval()

transform = transforms.Compose([
    transforms.Resize((xceptionnet_size, xceptionnet_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

data = CSVDataset(os.getcwd() + '/data/images/', os.getcwd() + '/data/splits/test.csv', 'image_id', 'deepfake', transform=transform)
dataloader = DataLoader(data)

all_labels = []
all_predictions = []
for i, data_example in enumerate(tqdm(dataloader)):
    (inputs, labels), name = data_example
    current_video = name[0].rsplit('_frame', 1)[0]
    label = labels.data[0].item()
    all_labels.append(label)
    output = xceptionnet(inputs)
    score = F.softmax(output, dim=1)[:, 1].cpu().data.numpy()
    pred = int(round(float(score)))
    all_predictions.append(pred)
    print(label == pred)

xceptionnet_member = Member(name="XceptionNet", train_probs=all_predictions, train_classes=all_labels, val_probs=all_predictions, val_classes=all_labels, submission_probs=all_predictions)

# Initialise ensemble
stack = StackEnsemble()
members = [capsule_member, dsp_fwa_member, ictu_oculi_member, xceptionnet_member, mantranet_member]
stack.add_members(members)

stack.fit()
stack.describe()