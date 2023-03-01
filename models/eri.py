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
        self.snippet_size = args['snippet_size']
        self.optim_type = args['optimizer']
        self.scheduler_type = args['lr_scheduler']
        self.gamma = args['lr_decay_rate']
        self.decay_steps = args['lr_decay_steps']
        self.epochs = args['num_epochs']

        self.model = getattr(models, args['model_name'])()

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model.out_c, dim_feedforward=256, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.head = nn.Linear(self.model.out_c, 7, bias=False)
    
    def forward(self, x):
        b, n, c, h, w = x.shape

        x = self.model(x.view(b*n, c, h, w))
        x = self.transformer(x.view(b, n, -1))
        x = torch.mean(x, dim=1)
        x = torch.sigmoid(self.head(x))

        return x

    def configure_optimizers(self):
        if self.optim_type == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optim_type == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        if self.scheduler_type == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_steps, gamma=self.gamma)
        elif self.scheduler_type == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        elif self.scheduler_type == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
            
        return [optimizer], [lr_scheduler]
    
    def _calculate_loss(self, batch, mode="train"):
        data, labels = batch
        imgs = data['images'].to(self.device)
        labels = labels.to(self.device)
        preds = self(imgs)
        # loss = F.mse_loss(preds, labels)
        loss = torch.mean(torch.abs(preds - labels))
        # print(loss)
        return loss

    def training_step(self, batch, batch_idx):
        # TODO: add logging for each step, also calculate epoch loss in training_epoch_end
        loss = self._calculate_loss(batch, mode="train")

        # self.log("train_a", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        
        return loss


    def validation_step(self, batch, batch_idx):
        vid_preds = {}
        data, labels = batch
        imgs = data['images'].to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            preds = self(imgs)
        result = {"val_preds": preds,
                  "val_labels": labels}
        return result
    
    def validation_epoch_end(self, validation_step_outputs):

        preds = torch.stack([data['val_preds'] for data in validation_step_outputs])
        labels = torch.stack([data['val_labels'] for data in validation_step_outputs])

        preds_mean = torch.mean(preds, dim=0, keepdim=True)
        labels_mean = torch.mean(labels, dim=0, keepdim=True)

        pcc = torch.sum((preds-preds_mean) * (labels-labels_mean), dim=0) / \
            (torch.sum((preds-preds_mean)**2) * torch.sum((labels-labels_mean)**2))**0.5
        
        apcc = torch.mean(pcc)

        self.log('val_apcc', apcc, on_epoch=True)
        result = {"val_apcc": apcc}
        # print(result)
        return result

    def test_step(self, batch, batch_idx):
        pass


if __name__ == '__main__':

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ERI(model_name="Res50", lr=1e-4, snippet_size=30)#.cuda()

    x = {}
    x['images'] = torch.rand(4, 30, 3, 256, 256)
    y = torch.rand(4, 7)
    loss = model._calculate_loss((x, y))

    # yy = model(x)
    print(y.shape)
