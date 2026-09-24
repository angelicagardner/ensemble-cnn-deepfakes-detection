"""
In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
IEEE International Workshop on Information Forensics and Security (WIFS), 2018
Yuezun Li, Ming-ching Chang and Siwei Lyu

The original implementation is written in TensorFlow:
https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi/blob/master/blink_net.py (BlinkCNN)
https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi/blob/master/deep_base/vgg16.py (network structure)

This file is a PyTorch port of the BlinkCNN (CNN-VGG16) network so it can be trained
together with the other single models. The layer structure follows deep_base/vgg16.py
and the pre-trained TensorFlow checkpoint provided by the authors is loaded directly.
"""

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# (layer name, number of output channels), 'pool' = 2x2 max pooling with stride 2
VGG16_LAYERS = [
    ("conv1_1", 64),
    ("conv1_2", 64),
    "pool",
    ("conv2_1", 128),
    ("conv2_2", 128),
    "pool",
    ("conv3_1", 256),
    ("conv3_2", 256),
    ("conv3_3", 256),
    "pool",
    ("conv4_1", 512),
    ("conv4_2", 512),
    ("conv4_3", 512),
    "pool",
    ("conv5_1", 512),
    ("conv5_2", 512),
    ("conv5_3", 512),
    "pool",
]


class BlinkCNN(nn.Module):
    """
    VGG16 network as defined in deep_base/vgg16.py (get_prob), with 2 output classes.
    """

    def __init__(self, num_class=2):
        super(BlinkCNN, self).__init__()
        self.conv_names = []
        self.layer_order = []
        in_channels = 3
        for layer in VGG16_LAYERS:
            if layer == "pool":
                self.layer_order.append("pool")
            else:
                name, out_channels = layer
                setattr(
                    self,
                    name,
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                )
                self.conv_names.append(name)
                self.layer_order.append(name)
                in_channels = out_channels
        self.fc6 = nn.Linear(7 * 7 * 512, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_class)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        for layer in self.layer_order:
            if layer == "pool":
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            else:
                x = F.relu(getattr(self, layer)(x))
        # TensorFlow flattens in NHWC order, keep that order so the original fc6 weights apply
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        # As in the original: fully-connected -> dropout (training only) -> relu
        x = F.relu(self.dropout(self.fc6(x)))
        x = F.relu(self.dropout(self.fc7(x)))
        return self.fc8(x)


# =============================================================================
# Experiment wrapper, binary (real/fake) output as used in the degree project.
# =============================================================================
class Ictu_Oculi(nn.Module):
    """Ictu Oculi (CNN-VGG16) with binary output (0 = real, 1 = deepfake)."""

    input_size = [3, 224, 224]
    # The original network takes BGR pixel values in the range 0-255 without mean subtraction
    # (PIXEL_MEAN is disabled in blink_cnn.yml). ToTensor() scales to 0-1, so std = 1/255
    # brings the values back to 0-255, and forward() swaps RGB to BGR.
    mean = [0.0, 0.0, 0.0]
    std = [1 / 255.0, 1 / 255.0, 1 / 255.0]

    # Folder (inside the pre-trained models folder) with the authors' TensorFlow checkpoint (ckpt_CNN)
    pretrained_folder = "ictu_oculi"

    def __init__(self):
        super(Ictu_Oculi, self).__init__()
        self.net = BlinkCNN(num_class=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x[:, [2, 1, 0], :, :]  # RGB -> BGR
        return self.net(x)

    def loss(self, outputs, labels):
        return self.criterion(outputs, labels)

    def fake_probability(self, outputs):
        return F.softmax(outputs, dim=1)[:, 1]

    def trainable_parameters(self):
        return self.parameters()

    def load_pretrained(self, folder):
        self.net.load_state_dict(
            _convert_tf_checkpoint(
                os.path.join(folder, self.pretrained_folder), self.net
            )
        )


def _convert_tf_checkpoint(folder, net):
    """Reads the authors' TensorFlow checkpoint and converts the weights to PyTorch layout."""
    import tensorflow as tf  # only needed to read the original checkpoint

    checkpoint = tf.train.latest_checkpoint(folder)
    if checkpoint is None:
        index_files = sorted(
            glob.glob(os.path.join(folder, "**", "*.index"), recursive=True)
        )
        if not index_files:
            raise FileNotFoundError(
                "No TensorFlow checkpoint found in {}".format(folder)
            )
        checkpoint = index_files[-1][: -len(".index")]
    reader = tf.train.load_checkpoint(checkpoint)
    variables = [
        name
        for name in reader.get_variable_to_shape_map()
        if not any(
            slot in name for slot in ("Momentum", "Adam", "RMSProp", "global_step")
        )
    ]

    def tf_variable(layer, kind):
        matches = [
            name
            for name in variables
            if name == "{}/{}".format(layer, kind)
            or name.endswith("/{}/{}".format(layer, kind))
        ]
        if len(matches) != 1:
            raise KeyError(
                "Expected one variable for {}/{} in checkpoint, found {}".format(
                    layer, kind, matches
                )
            )
        return reader.get_tensor(matches[0])

    state_dict = {}
    for layer in net.conv_names:
        # TensorFlow conv kernel [h, w, in, out] -> PyTorch [out, in, h, w]
        state_dict[layer + ".weight"] = torch.from_numpy(
            np.transpose(tf_variable(layer, "weights"), (3, 2, 0, 1)).copy()
        )
        state_dict[layer + ".bias"] = torch.from_numpy(tf_variable(layer, "biases"))
    for layer in ("fc6", "fc7", "fc8"):
        # TensorFlow dense kernel [in, out] -> PyTorch [out, in]
        state_dict[layer + ".weight"] = torch.from_numpy(
            np.transpose(tf_variable(layer, "weights")).copy()
        )
        state_dict[layer + ".bias"] = torch.from_numpy(tf_variable(layer, "biases"))
    return state_dict
