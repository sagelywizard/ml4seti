import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

class DenseNet(nn.Module):
    def __init__(self, pretrained):
        super(DenseNet, self).__init__()
        self.densenet = list(torchvision.models.densenet201(
            pretrained=pretrained
        ).features.named_children())
        self.densenet[0] = ('conv0', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.densenet = nn.Sequential(OrderedDict(self.densenet))
        self.linear = nn.Linear(3840, 7)

    def forward(self, minibatch):
        dense = self.densenet(minibatch)
        out = F.relu(dense, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(dense.size(0), -1)
        return self.linear(out)
