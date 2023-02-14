import torch
from torch import nn 
from torchvision.models import resnet50

class Res50(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        net = resnet50(pretrained=True)
        # need vggface2 pretrained model
        self.backbone = nn.Sequential(*list(net.children())[:-1])

        self.head = nn.Linear(2048, 7)

    def forward(self, x):
        x = self.backbone(x)
        y = torch.sigmoid(self.head(x.flatten(1)))
        return y


if __name__ == '__main__':
    
    net = Res50()

    x = torch.rand(4, 3, 224, 224)
    y = net(x)

    print(y.shape)
    print(y)