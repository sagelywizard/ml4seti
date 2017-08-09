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
        ).features
        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7)
        )
        self.linear = nn.Linear(3840, 7)

    def forward(self, minibatch):
        dense = self.densenet(self.conv(minibatch))
        out = self.out(dense)
        return self.linear(out.view(dense.size(0), -1))
