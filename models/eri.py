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
        elif 'res18' in self.features:
            # self.model = torch.nn.Identity()
            feat_ch = 512
        elif 'resnet50' in self.features:
            # self.model = torch.nn.Identity()
            feat_ch = 2048

        hidden_ch = 256
        self.conv_module = nn.Sequential(
            nn.Conv1d(feat_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.elem_atten = nn.Sequential(
            nn.Conv1d(hidden_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=-1),
        )
        feat_ch = hidden_ch

        self.tokens = 1
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.snippet_size + self.tokens, feat_ch))
        self.reg_token = nn.Parameter(torch.randn(1, 1, feat_ch))

        self.n_head = 4
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_ch, dim_feedforward=512, nhead=self.n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # self.layer_norm = nn.LayerNorm(feat_ch)

        # self.head = nn.Linear(feat_ch + 8, 7, bias=True)
        # self.head = nn.Linear(feat_ch, 7)

        self.head = nn.Sequential(
            nn.Linear(feat_ch * 2, hidden_ch, bias=False),
            nn.LayerNorm(hidden_ch),
            nn.Dropout(0.2),
            nn.Linear(hidden_ch, 7, bias=False),
        )

    def configure_optimizers(self):

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # params =[
        #     {'params': self.transformer.parameters(), 'lr': self.lr},
        #     {'params': self.head.parameters(), 'lr': self.lr}
        # ]

        if self.optim_type == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        elif self.optim_type == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim_type == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0)
            
        if self.scheduler_type == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_steps, gamma=self.gamma)
        elif self.scheduler_type == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        elif self.scheduler_type == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
            
        return [optimizer], [lr_scheduler]

    def forward_model_mask(self, data):
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
        
        reg_token = torch.tile(self.reg_token, (b, self.tokens, 1))
        x = torch.cat([reg_token, x], dim=1)
        x = self.transformer(x.permute(1, 0, 2), mask=mask).permute(1, 0, 2)

        x = x[:, 0]
        # x = torch.cat([x, age_con], dim=-1)
        preds = torch.sigmoid(self.head(x))

        return preds

    def forward_model_seq(self, data):
        input = data['images']
        age_con = data['age_con'].to(self.device)

        feats = []
        for i in range(len(input)):
            x = input[i].to(self.device)
            if self.features == 'image':
                n, c, h, w = x.shape
                x = self.model(x.view(n, c, h, w)).view(n, -1)
            else:
                n, c = x.shape

            x = x.reshape(1, n, -1)

            x = x.permute(0, 2, 1)
            x = self.conv_module(x)
            attn = self.elem_atten(x)
            # x = torch.sum(x * attn, dim=-1)
            x = x.permute(0, 2, 1)
            # print(x.shape)

            
            reg_token = self.reg_token
            x = torch.cat([reg_token, x], dim=1)
            x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)

            x1 = x[:, 0]
            x2 = torch.sum(x[:, 1:] * attn.permute(0, 2, 1), dim=1)

            x = torch.cat([x1, x2], dim=-1)
            feats.append(x)

        feats = torch.cat(feats, dim = 0)
        preds = torch.sigmoid(self.head(feats))

        return preds
    
    def forward_model(self, data):
        
        if self.snippet_size > 0:
            preds = self.forward_model_mask(data)
        else:
            preds = self.forward_model_seq(data)

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
        
        # # hard samples
        # quantile = torch.quantile(losses, 0.5, interpolation='nearest')
        # mask[losses > quantile] = 2.

        # noisy samples
        quantile = torch.quantile(losses, 0.98, interpolation='nearest')
        mask[losses > quantile] = 0.5
        return mask
    
    def compute_loss(self, preds, labels):
        if self.loss_type == 'l2':
            loss_l2 = F.mse_loss(preds, labels, reduction='none')
            mask = self.hard_sample_mask(loss_l2, False)
            loss = torch.mean(loss_l2 * mask)
        elif self.loss_type == 'l1':
            loss_l1 = torch.abs(preds - labels)
            mask = self.hard_sample_mask(loss_l1, False)
            loss = torch.mean(loss_l1 * mask)
        elif self.loss_type == 'pcc':
            loss = 1 - self.calculate_apcc(preds, labels)
        
        return loss


    def training_step(self, batch, batch_idx):
        # TODO: add logging for each step, also calculate epoch loss in training_epoch_end
        # loss = self._calculate_loss(batch, mode="train")
        data, labels = batch
        preds = self.forward_model(data)
        loss = self.compute_loss(preds, labels)

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
        'pretrained': '',
        'features': 'smm',
        'loss': 'l2'

    }
    model = ERI(**args)

    x = {}
    x['images'] = torch.rand(4, 30, 3, 256, 256)
    y = torch.rand(4, 7)
    loss = model.forward_model(x)

    # yy = model(x)
    # print(y.shape)

    result = model.val_dataloader()
