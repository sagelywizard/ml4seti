import torch.nn as nn
import torchvision

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.densenet = torchvision.models.densenet201(
            pretrained=False,
            num_classes=7
        )

    def forward(self, minibatch):
        return self.densenet(self.conv(minibatch))
