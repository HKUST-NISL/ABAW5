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
        self.features = args['features']
        self.loss_type = args['loss']

        self.batch_size = args['batch_size']

        self.pretrained_path = args['pretrained']

        if self.features =='image':
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

            feat_ch = self.model.out_c
        elif 'smm' in self.features:
            # self.model = torch.nn.Identity()
            feat_ch = 272
        elif 'effnetb0' in self.features:
            # self.model = torch.nn.Identity()
            feat_ch = 1280
        elif 'res' in self.features:
            # self.model = torch.nn.Identity()
            feat_ch = 2048

        self.n_head = 4
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_ch, dim_feedforward=feat_ch, nhead=self.n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.head = nn.Sequential(
            nn.Linear(feat_ch + 8, 7, bias=True),
        )

        # self.head = nn.Linear(feat_ch, 7, bias=True)
        

    def configure_optimizers(self):

        # for param in self.model.parameters():
        #     param.requires_grad = False

        params =[
            {'params': self.transformer.parameters(), 'lr': self.lr},
            {'params': self.head.parameters(), 'lr': self.lr}
        ]

        if self.optim_type == 'adamw':
            optimizer = optim.AdamW(params, lr=self.lr, weight_decay=0.01)
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
    
    def forward_model(self, data):
        x = data['images'].to(self.device)
        age_con = data['age_con'].to(self.device)
        mask = data['mask'].to(self.device)
        mask = torch.cat([mask] * self.n_head, dim=0)

        if self.features == 'image':
            b, n, c, h, w = x.shape
            x = self.model(x.view(b*n, c, h, w)).view(b, n, -1)
        else:
            b, n, c = x.shape

        x = x.reshape(b, n, -1)
        mask_1d = mask[:b, :, :1]
        
        # x_mean = torch.sum(x * mask_1d, dim=1) / torch.sum(mask_1d, dim=1)
        # x = x - x_mean.unsqueeze(1)

        x = self.transformer(x.permute(1, 0, 2), mask=mask).permute(1, 0, 2)

        # print(mask_1d.shape, x.shape)
        x = torch.sum(x * mask_1d, dim=1) / torch.sum(mask_1d, dim=1)
        x = torch.cat([x, age_con], dim=1)
        preds = torch.sigmoid(self.head(x))
        return preds
    
    def calculate_apcc(self, preds, labels):
        preds_mean = torch.mean(preds, dim=0, keepdim=True)
        labels_mean = torch.mean(labels, dim=0, keepdim=True)

        pcc = torch.sum((preds-preds_mean) * (labels-labels_mean), dim=0) / \
            torch.clamp((torch.sum((preds-preds_mean)**2, dim=0) * torch.sum((labels-labels_mean)**2, dim=0))**0.5, min=1e-8)
    
        return torch.mean(pcc)
        
    def hard_sample_mask(self, losses, flag=True):
        mask = torch.ones(losses.shape).to(losses.device)
        if not flag:
            return mask

        quantile = torch.quantile(losses, 0.6, interpolation='linear')
        print(quantile, torch.median(losses))
        mask[losses > quantile] = 5
        return mask

    def training_step(self, batch, batch_idx):
        # TODO: add logging for each step, also calculate epoch loss in training_epoch_end
        # loss = self._calculate_loss(batch, mode="train")
        data, labels = batch
        preds = self.forward_model(data)

        if self.loss_type == 'l2':
            loss_l2 = F.mse_loss(preds, labels, reduce=False)
            mask = self.hard_sample_mask(loss_l2, False)
            loss = torch.mean(loss_l2 * mask)
        elif self.loss_type == 'l1':
            loss_l1 = torch.abs(preds, labels)
            mask = self.hard_sample_mask(loss_l1)
            loss = torch.mean(loss_l1 * mask)
        elif self.loss_type == 'pcc':
            loss = 1 - self.calculate_apcc(preds, labels)

        result = {"train_preds": preds,   
                  "train_labels": labels,
                  "loss": loss}
        return result
    
    def training_epoch_end(self, training_step_outputs):

        preds = torch.cat([data['train_preds'] for data in training_step_outputs], dim=0)
        labels = torch.cat([data['train_labels'] for data in training_step_outputs], dim=0)

        apcc = self.calculate_apcc(preds, labels)
        
        self.log('train_apcc', apcc, on_epoch=True, prog_bar=True)
        result = {"train_apcc": apcc}


    def validation_step(self, batch, batch_idx):
        data, labels = batch
        preds = self.forward_model(data)
        result = {"val_preds": preds,   
                  "val_labels": labels}
        return result
    
    def validation_epoch_end(self, validation_step_outputs):

        preds = torch.cat([data['val_preds'] for data in validation_step_outputs], dim=0)
        labels = torch.cat([data['val_labels'] for data in validation_step_outputs], dim=0)

        l1_loss = torch.mean(torch.abs(preds - labels))
        self.log('val_l1loss', l1_loss, on_epoch=True, prog_bar=True)

        apcc = self.calculate_apcc(preds, labels)
        
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
    model = ERI(**args)

    x = {}
    x['images'] = torch.rand(4, 30, 3, 256, 256)
    y = torch.rand(4, 7)
    loss = model._calculate_loss((x, y))

    # yy = model(x)
    # print(y.shape)

    result = model.val_dataloader()
