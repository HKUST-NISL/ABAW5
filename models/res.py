import torch
from torch import nn
import numpy as np

import torchvision

class ResNetEmo(nn.Module):
    def __init__(self, num_classes=8, version=18):
        super().__init__()

        if version == 18:
            resnet = torchvision.models.resnet18(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            # self.fc = nn.Linear(512, num_classes)
        elif version == 50:
            resnet = torchvision.models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            # self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):

        x = torch.flatten(self.backbone(x), 1)
        # x = self.fc(x)

        return x


if __name__ == '__main__':

    x = torch.rand(64, 3, 224, 224)

    model = ResNetEmo()

    y = model(x)

    print(y.shape)