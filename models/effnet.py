from turtle import forward
import torch
from torch import nn
import numpy as np

import torchvision
import timm


class EffNet(nn.Module):
    def __init__(self, model_type = 0):
        super().__init__()
        
        if model_type == 0:
            model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
            model.classifier = torch.nn.Identity()
            model.load_state_dict(torch.load('./pretrained/state_vggface2_enet0_new.pt', map_location='cpu')) #_new
            self.out_c = 1280
        else:
            model = timm.create_model('tf_efficientnet_b2_ns', pretrained=False)
            model.classifier=torch.nn.Identity()
            model.load_state_dict(torch.load('./pretrained/state_vggface2_enet2.pt', map_location='cpu'))
            self.out_c = 1408

        for param in model.parameters():
            param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True


        self.model = model

    def forward(self, x):

        y = self.model(x)

        return y
    

def effnetb0():
    return EffNet()

def effnetb2():
    return EffNet(2)


if __name__ == '__main__':

    net = effnetb0()#.cuda()

    x = torch.rand(64, 3, 224, 224)#.cuda()

    y = net(x)

    print(y.shape)