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
        self.sample_times = args['sample_times']

        self.optim_type = args['optimizer']
        self.scheduler_type = args['lr_scheduler']
        self.gamma = args['lr_decay_rate']
        self.decay_steps = args['lr_decay_steps']
        self.epochs = args['num_epochs']
        self.args = args

        if self.args['load_feature'] == 'False':
            self.pretrained = args['pretrained']
            self.model = getattr(models, args['model_name'])()
            ckpt = torch.load(self.pretrained, map_location=torch.device('cpu'))['state_dict']
            self.model.load_state_dict(ckpt)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.model.out_c, dim_feedforward=256, nhead=4)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.head = nn.Linear(self.model.out_c, 7, bias=False)
        else:
            #encoder_layer = nn.TransformerEncoderLayer(d_model=272, dim_feedforward=256, nhead=4)
            #self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
            #self.head = nn.Linear(272, 7, bias=False)
            encoder_layer = nn.TransformerEncoderLayer(d_model=272, dim_feedforward=512, nhead=8)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
            self.fc1 = nn.Linear(272, 64)
            self.head = nn.Linear(64, 7)
    
    def forward(self, x):
        if self.args['load_feature'] == 'False':
            b, n, c, h, w = x.shape # 4, 30, 3, 299, 299
            x = self.model(x.view(b*n, c, h, w))
        else:
            b, n, _ = x.shape
        x = self.transformer(x.view(b, n, -1))
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.head(x)) # 4, 7
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
        loss = F.mse_loss(preds, labels)
        #loss = torch.mean(torch.abs(preds - labels))
        # print(loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        # self.log("train_a", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        result = {'train_loss': loss}
        print(result)
        return loss

    def validation_step(self, batch, batch_idx):
        vid_preds = {}
        data, labels = batch
        imgs = data['images'].to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            preds = self(imgs)
        loss = F.mse_loss(preds, labels)
        result = {"val_preds": preds,
                  "val_labels": labels, "val_loss": loss}
        return result
    
    def validation_epoch_end(self, validation_step_outputs):

        preds = torch.cat([data['val_preds'] for data in validation_step_outputs], dim=0)
        labels = torch.cat([data['val_labels'] for data in validation_step_outputs], dim=0)
        loss = [data['loss'] for data in validation_step_outputs].mean()

        preds = torch.mean(preds.reshape(-1, self.sample_times, 7), dim=1)
        labels = torch.mean(labels.reshape(-1, self.sample_times, 7), dim=1)

        preds_mean = torch.mean(preds, dim=0, keepdim=True)
        labels_mean = torch.mean(labels, dim=0, keepdim=True)

        pcc = torch.sum((preds-preds_mean) * (labels-labels_mean), dim=0) / \
            (torch.sum((preds-preds_mean)**2, dim=0) * torch.sum((labels-labels_mean)**2, dim=0))**0.5
        
        apcc = torch.mean(pcc)

        self.log('val_apcc', apcc, on_epoch=True)
        result = {"val_apcc": apcc, "val_loss": loss}
        print(result)
        return result

    def test_step(self, batch, batch_idx):
        pass


if __name__ == '__main__':

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ERI(model_name="Res50", lr=1e-4, snippet_size=30, load_feature='True')#.cuda()

    x = {}
    x['images'] = torch.rand(4, 272) # torch.rand(4, 30, 3, 256, 256)
    y = torch.rand(4, 7)
    loss = model._calculate_loss((x, y))
    # yy = model(x)
    print(y.shape)
