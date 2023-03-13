import os
import numpy as np
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
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

        if self.args['load_feature'] == 'False':
            self.pretrained = args['pretrained']
            self.model = getattr(models, args['model_name'])()
            ckpt = torch.load(self.pretrained, map_location=torch.device('cpu'))['state_dict']
            self.model.load_state_dict(ckpt)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.model.out_c, dim_feedforward=256, nhead=4)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.head = nn.Linear(self.model.out_c, 7, bias=False)
        elif self.args['load_feature'] == 'smm':
            # for first model: US
            encoder_layer = nn.TransformerEncoderLayer(d_model=272, dim_feedforward=256, nhead=4)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.head = nn.Linear(272, 7, bias=False)
            if self.args['two_models'] == 'True':
                # for second model: SA
                encoder_layer2 = nn.TransformerEncoderLayer(d_model=272, dim_feedforward=256, nhead=4)
                self.transformer2 = nn.TransformerEncoder(encoder_layer2, num_layers=6)
                self.head2 = nn.Linear(272, 7, bias=False)
                self.transformer = [self.transformer1, self.transformer2]
                self.head = [self.head1, self.head2]
        elif self.args['load_feature'] == 'vgg':
            self.linear1 = nn.Linear(2048, 128)
            self.linear2 = nn.Linear(128, 7)
        else:
            print('load feature should be one of the followings: [False, smm, vgg]')

    def sigmoid_(self, x):
        return 1 / (1 + torch.exp(-10*x))

    def forward(self, x, country, age):
        if self.args['load_feature'] == 'False':
            x = x.to(self.device)
            b, n, c, h, w = x.shape # 4, 30, 3, 299, 299
            x = self.model(x.view(b*n, c, h, w))
            x = self.transformer(x.view(b, n, -1))
            x = torch.mean(x, dim=1)
            x = torch.sigmoid(self.head(x))  # 4, 7
            return x
        elif self.args['load_feature'] == 'smm':
            outputs = []
            b = len(x)
            for i in range(b):
                input = x[i].to(self.device) #.unsqueeze(0)
                if self.args['two_models'] == 'False':
                    input = self.transformer(input)
                    input = torch.mean(input, dim=0)
                    #input = self.head1(input)
                    #input = self.sigmoid_(self.head1(input))
                    input = torch.sigmoid(self.head(input))
                else:
                    input = self.transformer[country[i]](input)
                    input = torch.mean(input, dim=0)
                    input = self.head[country[i]](input)
                outputs.append(input)
            outputs = torch.stack(outputs)
            return outputs
        elif self.args['load_feature'] == 'vgg':
            outputs = []
            b = len(x)
            for i in range(b):
                input = x[i].to(self.device).unsqueeze(0) #1, 300, 2048
                input = F.relu(self.linear1(input))
                input = self.linear2(input)  # 1, 300, 7
                input = torch.mean(input, dim=1) # 1, 7
                input = torch.sigmoid(input)
                outputs.append(input)
            outputs = torch.stack(outputs)
            return outputs

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
    
    def _calculate_loss(self, batch, train):
        data, labels = batch
        imgs = data['images'] #.to(self.device)
        age = data['age'].to(self.device)
        country = data['country'].to(self.device)
        labels = labels.to(self.device)
        preds = self(imgs, country, age)
        #loss = F.mse_loss(preds, labels)
        #loss = torch.mean(torch.abs(preds - labels))
        # print(loss)
        loss = self.pcc_loss(preds, labels, train)
        '''labels = (labels > 0.5).int().float()
        loss = self.bce_loss(preds, labels)'''
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, True)
        # self.log("train_a", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        result = {'loss': loss}
        return result

    def training_epoch_end(self, training_step_outputs):
        loss = np.mean([data['loss'].item() for data in training_step_outputs])
        print('training loss:', loss)

    def validation_step(self, batch, batch_idx):
        vid_preds = {}
        data, labels = batch
        imgs = data['images'] #.to(self.device)
        age = data['age'].to(self.device)
        country = data['country'].to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            preds = self(imgs, country, age)
        #loss = F.mse_loss(preds, labels)
        loss = self.pcc_loss(preds, labels, False)
        '''new_labels = (labels > 0.5).int().float()
        loss = self.bce_loss(preds, new_labels)
        new_preds = (preds > 0.5).int().float()'''
        result = {"val_preds": preds,
                  "val_labels": labels, "val_loss": loss.item()}
        return result

    '''def ccc(gold, pred):
        gold = K.squeeze(gold, axis=-1)
        pred = K.squeeze(pred, axis=-1)
        gold_mean = K.mean(gold, axis=-1, keepdims=True)
        pred_mean = K.mean(pred, axis=-1, keepdims=True)
        covariance = (gold - gold_mean) * (pred - pred_mean)
        gold_var = K.mean(K.square(gold - gold_mean), axis=-1, keepdims=True)
        pred_var = K.mean(K.square(pred - pred_mean), axis=-1, keepdims=True)
        ccc = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
        return ccc'''

    def pcc_loss_modified(self, preds, labels):
        preds = torch.mean(preds.reshape(-1, self.sample_times, 7), dim=1)
        labels = torch.mean(labels.reshape(-1, self.sample_times, 7), dim=1)

        preds_mean = torch.mean(preds, dim=0, keepdim=True)
        labels_mean = torch.mean(labels, dim=0, keepdim=True)
        covariance = torch.sum((preds - preds_mean) * (labels - labels_mean), dim=0)
        preds_var = torch.sum((preds - preds_mean) ** 2, dim=0)
        labels_var = torch.sum((labels - labels_mean) ** 2,dim=0)

        pcc = covariance / ((preds_var * labels_var) ** 0.5 + 1e-7)

        apcc = torch.mean(pcc)
        loss = 1 - torch.mean(apcc)
        return loss

    def pcc_loss(self, preds, labels, train):
        pcc = self.pcc(preds, labels, train)
        loss = 1 - torch.mean(pcc)
        return loss

    def pcc(self, preds, labels, train):
        preds = torch.mean(preds.reshape(-1, self.sample_times, 7), dim=1)
        labels = torch.mean(labels.reshape(-1, self.sample_times, 7), dim=1)

        preds_mean = torch.mean(preds, dim=0, keepdim=True)
        labels_mean = torch.mean(labels, dim=0, keepdim=True)
        if train:
            pcc = torch.sum((preds - preds_mean) * (labels - labels_mean), dim=0) / \
                  ((torch.sum((preds - preds_mean) ** 2, dim=0) * torch.sum((labels - labels_mean) ** 2, dim=0)) ** 0.5 + (
                      1e-7))
        else:
            pcc = torch.sum((preds - preds_mean) * (labels - labels_mean), dim=0) / \
                  ((torch.sum((preds - preds_mean) ** 2, dim=0) * torch.sum((labels - labels_mean) ** 2,
                                                                            dim=0)) ** 0.5)
        return torch.mean(pcc)
    
    def validation_epoch_end(self, validation_step_outputs):
        preds = torch.cat([data['val_preds'] for data in validation_step_outputs], dim=0)
        labels = torch.cat([data['val_labels'] for data in validation_step_outputs], dim=0)
        # unnorm the preds
        #preds = preds * 0.3592 + 0.3652
        #labels = labels * 0.3592 + 0.3652

        loss = np.mean([data['val_loss'] for data in validation_step_outputs])
        apcc = self.pcc(preds, labels, False)

        self.log('val_apcc', apcc, on_epoch=True)
        result = {"val_apcc": apcc, "val_loss": loss}
        print(result)
        return result

    def test_step(self, batch, batch_idx):
        data, labels = batch
        imgs = data['images']  # .to(self.device)
        age = data['age'].to(self.device)
        country = data['country'].to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            preds = self(imgs, country, age)
        result = {"test_preds": preds,
                  "test_labels": labels,
                  "test_vid": data['vid']}
        return result

    def test_epoch_end(self, validation_step_outputs):
        preds = torch.cat([data['test_preds'] for data in validation_step_outputs], dim=0)
        labels = torch.cat([data['test_labels'] for data in validation_step_outputs], dim=0)
        vids = torch.cat([data['test_vid'] for data in validation_step_outputs], dim=0)
        # unnorm the preds
        #preds = preds * 0.3592 + 0.3652
        #labels = labels * 0.3592 + 0.3652

        apcc = self.pcc(preds, labels, False)
        self.log('test_apcc', apcc, on_epoch=True)
        result = {"test_apcc": apcc}
        print(result)
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        vids = vids.detach().cpu().numpy()
        np.save('preds_train.npy', preds)
        np.save('labels_train.npy', labels)
        np.save('vids_val.npy', vids)
        return result


if __name__ == '__main__':

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ERI(model_name="Res50", lr=1e-4, snippet_size=30, load_feature='True')#.cuda()

    x = {}
    x['images'] = torch.rand(4, 272) # torch.rand(4, 30, 3, 256, 256)
    y = torch.rand(4, 7)
    loss = model._calculate_loss((x, y))
    # yy = model(x)
    print(y.shape)
