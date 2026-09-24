import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import os, math


class ResNet(nn.Module):
    def __init__(self, layers=18, num_class=2, pretrained=True):
        super(ResNet, self).__init__()
        if layers == 18:
            self.resnet = models.resnet18(pretrained=pretrained)
        elif layers == 34:
            self.resnet = models.resnet34(pretrained=pretrained)
        elif layers == 50:
            self.resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            self.resnet = models.resnet101(pretrained=pretrained)
        elif layers == 152:
            self.resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError("layers should be 18, 34, 50, 101.")
        self.num_class = num_class
        if layers in [18, 34]:
            self.fc = nn.Linear(512, num_class)
        if layers in [50, 101, 152]:
            self.fc = nn.Linear(512 * 4, num_class)

    def conv_base(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)
        return layer1, layer2, layer3, layer4

    def forward(self, x):
        layer1, layer2, layer3, layer4 = self.conv_base(x)
        x = self.resnet.avgpool(layer4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SPPNet(nn.Module):
    def __init__(self, backbone=101, num_class=2, pool_size=(1, 2, 6), pretrained=True):
        # Only resnet is supported in this version
        super(SPPNet, self).__init__()
        if backbone in [18, 34, 50, 101, 152]:
            self.resnet = ResNet(backbone, num_class, pretrained)
        else:
            raise ValueError("Resnet{} is not supported yet.".format(backbone))

        if backbone in [18, 34]:
            self.c = 512
        if backbone in [50, 101, 152]:
            self.c = 2048

        self.spp = SpatialPyramidPool2D(out_side=pool_size)
        num_features = self.c * (
            pool_size[0] ** 2 + pool_size[1] ** 2 + pool_size[2] ** 2
        )
        self.classifier = nn.Linear(num_features, num_class)

    def forward(self, x):
        _, _, _, x = self.resnet.conv_base(x)
        x = self.spp(x)
        x = self.classifier(x)
        return x


class SpatialPyramidPool2D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        # batch_size, c, h, w = x.size()
        out = None
        for n in self.out_side:
            w_r, h_r = map(
                lambda s: math.ceil(s / n), x.size()[2:]
            )  # Receptive Field Size
            s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
            max_pool = nn.MaxPool2d(kernel_size=(w_r, h_r), stride=(s_w, s_h))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out


# =============================================================================
# Experiment wrapper (not part of the original DSP-FWA code above).
# SPPNet with a ResNet50 backbone and binary (real/fake) output, as used in the
# degree project.
# =============================================================================
class DSP_FWA(nn.Module):
    """DSP-FWA (SPPNet, ResNet50 backbone) with binary output (0 = real, 1 = deepfake)."""

    input_size = [3, 224, 224]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Pre-trained checkpoint provided by the original authors
    pretrained_file = "SPP-res50.pth"

    def __init__(self):
        super(DSP_FWA, self).__init__()
        # ImageNet weights are not needed here, all weights come from the authors' checkpoint
        self.net = SPPNet(backbone=50, num_class=2, pretrained=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def loss(self, outputs, labels):
        return self.criterion(outputs, labels)

    def fake_probability(self, outputs):
        return F.softmax(outputs, dim=1)[:, 1]

    def trainable_parameters(self):
        return self.parameters()

    def load_pretrained(self, folder):
        checkpoint = torch.load(
            os.path.join(folder, self.pretrained_file), map_location="cpu"
        )
        state_dict = checkpoint["net"] if "net" in checkpoint else checkpoint
        state_dict = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        self.net.load_state_dict(state_dict)
