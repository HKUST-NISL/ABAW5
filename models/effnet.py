from turtle import forward
import torch
from torch import nn
import numpy as np

import torchvision
import timm


class EffNetEmo(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
        model.classifier=torch.nn.Identity()
        model.load_state_dict(torch.load('./pretrained/state_vggface2_enet0_new.pt')) #_new
        model.classifier=nn.Sequential(nn.Linear(in_features=1280, out_features=num_classes)) #1792 #1280 #1536

        # effnet b2
        # model = timm.create_model('tf_efficientnet_b2_ns', pretrained=False)
        # model.classifier=torch.nn.Identity()
        # model.load_state_dict(torch.load('./models/state_vggface2_enet0_new.pt')) #_new
        # model.load_state_dict(torch.load('./models/state_vggface2_enet2.pt'))
        # model.classifier=nn.Sequential(nn.Linear(in_features=1408, out_features=num_classes)) #1792 #1280 #1536

        # model = torch.load('../face-emotion-recognition/models/affectnet_emotions/enet_b2_8.pt')

        for param in model.parameters():
            param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True


        self.model = model

    def forward(self, x):

        y = self.model(x)

        return y


if __name__ == '__main__':

    net = EffNetEmo().cuda()

    x = torch.rand(64, 3, 224, 224).cuda()

    y = net(x)

    print(y.shape)