"""
Copyright (c) 2018, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for Capsule-Forensics model
"""

import sys

sys.setrecursionlimit(15000)
import torch
import torch.nn.functional as F
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as models

NO_CAPS = 10


class StatsNet(nn.Module):
    def __init__(self):
        super(StatsNet, self).__init__()

    def forward(self, x):
        x = x.view(x.data.shape[0], x.data.shape[1], x.data.shape[2] * x.data.shape[3])

        mean = torch.mean(x, 2)
        std = torch.std(x, 2)

        return torch.stack((mean, std), dim=1)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class VggExtractor(nn.Module):
    def __init__(self, train=False):
        super(VggExtractor, self).__init__()

        self.vgg_1 = self.Vgg(models.vgg19(pretrained=True), 0, 18)
        if train:
            self.vgg_1.train(mode=True)
            self.freeze_gradient()
        else:
            self.vgg_1.eval()

    def Vgg(self, vgg, begin, end):
        features = nn.Sequential(*list(vgg.features.children())[begin : (end + 1)])
        return features

    def freeze_gradient(self, begin=0, end=9):
        for i in range(begin, end + 1):
            self.vgg_1[i].requires_grad = False

    def forward(self, input):
        return self.vgg_1(input)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.capsules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    StatsNet(),
                    nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(8),
                    nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(1),
                    View(-1, 8),
                )
                for _ in range(NO_CAPS)
            ]
        )

    def squash(self, tensor, dim):
        squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm))

    def forward(self, x):
        # outputs = [capsule(x.detach()) for capsule in self.capsules]
        # outputs = [capsule(x.clone()) for capsule in self.capsules]
        outputs = [capsule(x) for capsule in self.capsules]
        output = torch.stack(outputs, dim=-1)

        return self.squash(output, dim=-1)


class RoutingLayer(nn.Module):
    def __init__(
        self,
        gpu_id,
        num_input_capsules,
        num_output_capsules,
        data_in,
        data_out,
        num_iterations,
    ):
        super(RoutingLayer, self).__init__()

        self.gpu_id = gpu_id
        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(
            torch.randn(num_output_capsules, num_input_capsules, data_out, data_in)
        )

    def squash(self, tensor, dim):
        squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm))

    def forward(self, x, random, dropout):
        # x[b, data, in_caps]

        x = x.transpose(2, 1)
        # x[b, in_caps, data]

        if random:
            noise = Variable(0.01 * torch.randn(*self.route_weights.size()))
            if self.gpu_id >= 0:
                noise = noise.cuda(self.gpu_id)
            route_weights = self.route_weights + noise
        else:
            route_weights = self.route_weights

        priors = route_weights[:, None, :, :, :] @ x[None, :, :, :, None]

        # route_weights [out_caps , 1 , in_caps , data_out , data_in]
        # x             [   1     , b , in_caps , data_in ,    1    ]
        # priors        [out_caps , b , in_caps , data_out,    1    ]

        priors = priors.transpose(1, 0)
        # priors[b, out_caps, in_caps, data_out, 1]

        if dropout > 0.0:
            drop = Variable(torch.FloatTensor(*priors.size()).bernoulli(1.0 - dropout))
            if self.gpu_id >= 0:
                drop = drop.cuda(self.gpu_id)
            priors = priors * drop

        logits = Variable(torch.zeros(*priors.size()))
        # logits[b, out_caps, in_caps, data_out, 1]

        if self.gpu_id >= 0:
            logits = logits.cuda(self.gpu_id)

        num_iterations = self.num_iterations

        for i in range(num_iterations):
            probs = F.softmax(logits, dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True), dim=3)

            if i != self.num_iterations - 1:
                delta_logits = priors * outputs
                logits = logits + delta_logits

        # outputs[b, out_caps, 1, data_out, 1]
        outputs = outputs.squeeze()

        if len(outputs.shape) == 3:
            outputs = outputs.transpose(2, 1).contiguous()
        else:
            outputs = outputs.unsqueeze_(dim=0).transpose(2, 1).contiguous()
        # outputs[b, data_out, out_caps]

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, num_class, gpu_id):
        super(CapsuleNet, self).__init__()

        self.num_class = num_class
        self.fea_ext = FeatureExtractor()
        self.fea_ext.apply(self.weights_init)

        self.routing_stats = RoutingLayer(
            gpu_id=gpu_id,
            num_input_capsules=NO_CAPS,
            num_output_capsules=num_class,
            data_in=8,
            data_out=4,
            num_iterations=2,
        )

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x, random=False, dropout=0.0):

        z = self.fea_ext(x)
        z = self.routing_stats(z, random, dropout=dropout)
        # z[b, data, out_caps]

        # classes = F.softmax(z, dim=-1)

        # class_ = classes.detach()
        # class_ = class_.mean(dim=1)

        # return classes, class_

        classes = F.softmax(z, dim=-1)
        class_ = classes.detach()
        class_ = class_.mean(dim=1)

        return z, class_


class CapsuleLoss(nn.Module):
    def __init__(self, gpu_id):
        super(CapsuleLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if gpu_id >= 0:
            self.cross_entropy_loss.cuda(gpu_id)

    def forward(self, classes, labels):
        loss_t = self.cross_entropy_loss(classes[:, 0, :], labels)

        for i in range(classes.size(1) - 1):
            loss_t = loss_t + self.cross_entropy_loss(classes[:, i + 1, :], labels)

        return loss_t


# =============================================================================
# Experiment wrapper (not part of the original Capsule-Forensics-v2 code above).
# Combines the VGG19 feature extractor and the capsule network into one model
# with a binary (real/fake) output, as used in the degree project.
# =============================================================================
import os


class Capsule(nn.Module):
    """Capsule-Forensics-v2 with binary output (0 = real, 1 = deepfake)."""

    # Input settings from the original training script (train_binary_ffpp.py)
    input_size = [3, 300, 300]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Pre-trained checkpoint provided by the original authors
    pretrained_file = "capsule_21.pt"

    def __init__(self):
        super(Capsule, self).__init__()
        gpu_id = 0 if torch.cuda.is_available() else -1
        self.vgg_ext = VggExtractor()
        self.capnet = CapsuleNet(2, gpu_id)
        self.capsule_loss = CapsuleLoss(gpu_id)

    def forward(self, x):
        x = self.vgg_ext(x)
        # Random routing noise and dropout are only used during training (as in the original code)
        classes, _ = self.capnet(
            x, random=self.training, dropout=0.05 if self.training else 0.0
        )
        return classes

    def loss(self, outputs, labels):
        # Cross-entropy loss summed over the output capsules (original CapsuleLoss)
        return self.capsule_loss(outputs, labels)

    def fake_probability(self, outputs):
        # Softmax output averaged over the capsules, probability of class 1 (deepfake)
        return F.softmax(outputs, dim=-1).mean(dim=1)[:, 1]

    def trainable_parameters(self):
        # The VGG19 feature extractor is frozen in the original implementation
        return self.capnet.parameters()

    def load_pretrained(self, folder):
        state_dict = torch.load(
            os.path.join(folder, self.pretrained_file), map_location="cpu"
        )
        self.capnet.load_state_dict(state_dict)
