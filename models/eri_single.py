import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr

class ERI_single(LightningModule):
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

        self.isLoadFeature = args['load_feature']

        if args['load_feature'] == 'smm':
            self.linear1 = nn.Linear(272, 256)
        elif args['load_feature'] == 'vgg':
            self.linear1 = nn.Linear(2048, 256)
        self.linear2 = nn.Linear(256, 64)
        self.head = nn.Linear(64, 7, bias=False)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
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
        imgs = data['image'].to(self.device)
        labels = labels.to(self.device)
        preds = self(imgs)
        #loss = F.mse_loss(preds, labels)
        #loss = torch.mean(torch.abs(preds - labels))
        loss = self.pcc_loss(preds, labels)
        # print(loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
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
        imgs = data['image'].to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            preds = self(imgs)
        result = {"val_preds": preds,
                  "val_labels": labels}
        return result

    def pcc_loss(self, preds, labels):
        preds = torch.mean(preds.reshape(-1, self.sample_times, 7), dim=1)
        labels = torch.mean(labels.reshape(-1, self.sample_times, 7), dim=1)

        preds_mean = torch.mean(preds, dim=0, keepdim=True)
        labels_mean = torch.mean(labels, dim=0, keepdim=True)

        pcc = torch.sum((preds - preds_mean) * (labels - labels_mean), dim=0) / \
              ((torch.sum((preds - preds_mean) ** 2, dim=0) * torch.sum((labels - labels_mean) ** 2, dim=0)) ** 0.5 + (1e-7))

        loss = 1 - torch.mean(pcc)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        preds = torch.cat([data['val_preds'] for data in validation_step_outputs], dim=0)
        labels = torch.cat([data['val_labels'] for data in validation_step_outputs], dim=0)

        loss = np.mean([data['val_loss'] for data in validation_step_outputs])

        preds = torch.mean(preds.reshape(-1, self.sample_times, 7), dim=1)
        labels = torch.mean(labels.reshape(-1, self.sample_times, 7), dim=1)

        preds_mean = torch.mean(preds, dim=0, keepdim=True)
        labels_mean = torch.mean(labels, dim=0, keepdim=True)

        pcc = torch.sum((preds - preds_mean) * (labels - labels_mean), dim=0) / \
              (torch.sum((preds - preds_mean) ** 2, dim=0) * torch.sum((labels - labels_mean) ** 2, dim=0)) ** 0.5

        apcc = torch.mean(pcc)

        self.log('val_apcc', apcc, on_epoch=True)
        result = {"val_apcc": apcc, "val_loss": loss}
        print(result)
        return result

    def test_step(self, batch, batch_idx):
        pass
