import os

import pandas as pd
import seaborn as sn
import torch
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
import models


class ERI(LightningModule):
    def __init__(self, **args):
        super().__init__()

        self.lr = args['lr']

        self.model = getattr(models, args['model_name'])()
        self.head = nn.Linear(self.model.out_c, 7, bias=False)
    
    def forward(self, x):
        return torch.sigmoid(self.head(self.model(x)))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1) #TODO: should be in args
        return [optimizer], [lr_scheduler]
    
    def _calculate_loss(self, batch, mode="train"):
        
        data, labels = batch
        imgs = data['images'].to(self.device).squeeze(1)
        labels = labels.to(self.device)
        preds = self(imgs)
        loss = F.mse_loss(preds, labels)
        return loss

    def training_step(self, batch, batch_idx):
        # TODO: add logging for each step, also calculate epoch loss in training_epoch_end
        loss = self._calculate_loss(batch, mode="train")
        
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: add logging, also add validation_epoch_end
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        pass


if __name__ == '__main__':

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ERI(model_name="Res50", lr=1e-4).cuda()

    x = {}
    x['images'] = torch.rand(4, 3, 256, 256)
    y = torch.rand(4, 7)
    loss = model._calculate_loss((x, y))
    print(y)
