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

        self.pretrained_path = args['pretrained']

        self.model = getattr(models, args['model_name'])()

        if len(self.pretrained_path) > 1:
            if '.pkl' in self.pretrained_path:
                import pickle
                with open(self.pretrained_path, 'rb') as f:
                    obj = f.read()
                ckpt = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
            else:
                ckpt = torch.load(self.pretrained_path)
                if 'state_dict' in ckpt.keys():
                    ckpt = ckpt['state_dict']
                # ckpt = torch.load(self.pretrained_path)['state_dict']

            self.model.load_state_dict(ckpt, strict=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model.out_c, dim_feedforward=self.model.out_c, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        n_hidden = 256
        self.head = nn.Sequential(
            nn.Linear(self.model.out_c + 8, 7, bias=True),
            # nn.ReLU6(inplace=True),
            # nn.Linear(n_hidden, 7, bias=False)
        )

    def configure_optimizers(self):
        # parameters = self.parameters()

        parameters = []
        parameters.extend(self.transformer.parameters())
        parameters.extend(self.head.parameters())

        for param in self.model.parameters():
            param.requires_grad = False

        # if self.optim_type == 'adamw':
        #     optimizer = optim.AdamW(parameters, lr=self.lr)
        # elif self.optim_type == 'adam':
        #     optimizer = optim.Adam(parameters, lr=self.lr)

        params =[
            # {'params': self.model.parameters(), 'lr': self.lr * 1.0},
            # {'params': self.model.parameters(), 'lr': self.lr*0.1},
            {'params': self.transformer.parameters(), 'lr': self.lr},
            {'params': self.head.parameters(), 'lr': self.lr}
        ]

        if self.optim_type == 'adamw':
            optimizer = optim.AdamW(params, lr=self.lr)
        elif self.optim_type == 'adam':
            optimizer = optim.Adam(params, lr=self.lr)
        elif self.optim_type == 'sgd':
            optimizer = optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=0.0)
            
        if self.scheduler_type == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_steps, gamma=self.gamma)
        elif self.scheduler_type == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        elif self.scheduler_type == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
            
        return [optimizer], [lr_scheduler]
    
    def forward_model(self, x, age_con):
        b, n, c, h, w = x.shape

        x = self.model(x.view(b*n, c, h, w)).view(b, n, -1)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x = self.transformer(x.view(b, n, -1))
        x = torch.mean(x, dim=1)#[0]
        x = torch.cat([x, age_con], dim=1)
        x = torch.sigmoid(self.head(x))

        # x = self.model(x.view(b*n, c, h, w)).view(b, n, -1)
        # # x_mean = torch.mean(x, dim=1, keepdim=True)
        # # # x = x - x_mean
        # x = self.transformer(x.view(b, n, -1)) 
        # x = torch.max(x, dim=1)[0] - torch.mean(x, dim=1)
        # x = torch.cat([x, age_con], dim=1)
        # x = torch.sigmoid(self.head(x))

        return x


    def training_step(self, batch, batch_idx):
        # TODO: add logging for each step, also calculate epoch loss in training_epoch_end
        # loss = self._calculate_loss(batch, mode="train")

        data, labels = batch
        imgs = data['images'].to(self.device)
        age_con = data['age_con'].to(self.device)
        labels = labels.to(self.device)
        preds = self.forward_model(imgs, age_con)
        # loss = F.mse_loss(preds, labels)
        # loss = torch.mean(torch.abs(preds - labels))

        preds_mean = torch.mean(preds, dim=0, keepdim=True)
        labels_mean = torch.mean(labels, dim=0, keepdim=True)

        pcc = torch.sum((preds-preds_mean) * (labels-labels_mean), dim=0) / \
            torch.clamp((torch.sum((preds-preds_mean)**2, dim=0) * torch.sum((labels-labels_mean)**2, dim=0))**0.5, min=1e-8)
        
        loss = 1 - torch.mean(pcc)
        # self.log("train_a", acc, on_step=False, on_epoch=True)
        result = {"train_preds": preds,   
                  "train_labels": labels,
                  "loss": loss}
        return result
    
    def training_epoch_end(self, training_step_outputs):

        preds = torch.cat([data['train_preds'] for data in training_step_outputs], dim=0)
        labels = torch.cat([data['train_labels'] for data in training_step_outputs], dim=0)

        preds_mean = torch.mean(preds, dim=0, keepdim=True)
        labels_mean = torch.mean(labels, dim=0, keepdim=True)

        pcc = torch.sum((preds-preds_mean) * (labels-labels_mean), dim=0) / \
            torch.clamp((torch.sum((preds-preds_mean)**2, dim=0) * torch.sum((labels-labels_mean)**2, dim=0))**0.5, min=1e-8)
        
        apcc = torch.mean(pcc)

        self.log('train_apcc', apcc, on_epoch=True, prog_bar=True)
        result = {"train_apcc": apcc}
        # print(result)
        # return result


    def validation_step(self, batch, batch_idx):
        vid_preds = {}
        data, labels = batch
        imgs = data['images'].to(self.device)
        age_con = data['age_con'].to(self.device)
        labels = labels.to(self.device)
        preds = self.forward_model(imgs, age_con)
        result = {"val_preds": preds,   
                  "val_labels": labels}
        return result
    
    def validation_epoch_end(self, validation_step_outputs):

        preds = torch.cat([data['val_preds'] for data in validation_step_outputs], dim=0)
        labels = torch.cat([data['val_labels'] for data in validation_step_outputs], dim=0)

        loss = torch.mean(torch.abs(preds - labels))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        preds_mean = torch.mean(preds, dim=0, keepdim=True)
        labels_mean = torch.mean(labels, dim=0, keepdim=True)

        pcc = torch.sum((preds-preds_mean) * (labels-labels_mean), dim=0) / \
            torch.clamp((torch.sum((preds-preds_mean)**2, dim=0) * torch.sum((labels-labels_mean)**2, dim=0))**0.5, min=1e-8)
        
        apcc = torch.mean(pcc)

        self.log('val_apcc', apcc, on_epoch=True, prog_bar=True)
        result = {"val_apcc": apcc}
        # print(result)
        return result

    def test_step(self, batch, batch_idx):
        pass


if __name__ == '__main__':

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = {
        'model_name': 'Res50',
        'lr': 1e-4,
        'snippet_size': 10,
        'sample_times': 5,
        'optimizer': 'adam',
        'lr_scheduler': 'cosine',
        'lr_decay_rate': 0.98,
        'lr_decay_steps': 10,
        'num_epochs': 100,
        'pretrained': ''
    }
    model = ERI(**args)#.cuda()

    x = {}
    x['images'] = torch.rand(4, 30, 3, 256, 256)
    y = torch.rand(4, 7)
    loss = model._calculate_loss((x, y))

    # yy = model(x)
    # print(y.shape)

    result = model.val_dataloader()
