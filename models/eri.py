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

from einops import rearrange, repeat


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')


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

        
        # self.conv_module = nn.Sequential(
        #     nn.Conv1d(feat_ch, hidden_ch, kernel_size=5, stride=1, padding=2, bias=False),
        #     nn.Conv1d(hidden_ch, hidden_ch, kernel_size=5, stride=1, padding=2, bias=False),
        #     nn.Conv1d(hidden_ch, hidden_ch, kernel_size=5, stride=1, padding=2, bias=False),
        #     # nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False),
        # )

        # feat_ch += 68*2
        hidden_ch = 256
        self.rnn = nn.GRU(feat_ch, hidden_ch, 2, batch_first=False)
        # self.rnn_lmk = nn.GRU(68*2, hidden_ch//2, 2, batch_first=False)
        # self.rnn = nn.LSTM(feat_ch, hidden_ch, 2, batch_first=False)

        # hidden_ch += hidden_ch//2

        # self.elem_atten = nn.Sequential(
        #     nn.Conv1d(hidden_ch, 1, kernel_size=1, stride=1, padding=0),
        #     nn.Softmax(dim=-1),
        # )
        
        self.tokens = 1
        # self.pos_embedding = nn.Parameter(torch.zeros(1, self.snippet_size + self.tokens, hidden_ch))
        # self.pos_embedding = sinusoidal_embedding(1000, hidden_ch)
        self.reg_token = nn.Parameter(torch.randn(1, self.tokens, hidden_ch))

        self.n_head = 4
        self.n_layers = 4
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_ch, dim_feedforward=256, nhead=self.n_head, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # self.head = nn.Linear(feat_ch, 7)
        self.head = nn.Sequential(
            nn.Linear(hidden_ch, 256, bias=False),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 7, bias=False),
        )

        # self.heads = nn.ModuleList([nn.Sequential(
        #     nn.Linear(hidden_ch, hidden_ch, bias=False),
        #     nn.LayerNorm(hidden_ch),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_ch, 1, bias=False),
        # ) for i in range(self.tokens)])

        self.all_params = []
        for name, param in self.named_parameters():
            self.all_params.append(param)

        self.test_vids = []
        self.exp_names = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']

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

        reg_token = torch.tile(self.reg_token, (b, 1, 1))
        x = torch.cat([reg_token, x], dim=1)
        x = x.reshape(b, n, -1) + self.pos_embedding.tile(b, 1, 1)[:, :n+1].to(x.device)

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

            # x = torch.cat([x, xlmk], dim=-1)
            x = x.reshape(1, n, -1)
            # xlmk = xlmk.reshape(1, n, -1)

            # x = x.permute(0, 2, 1)
            # x = self.conv_module(x)
            # attn = self.elem_atten(x)
            # # x = torch.sum(x * attn, dim=-1)
            # x = x.permute(0, 2, 1)
            # # print(x.shape)

            x, ho = self.rnn(x.permute(1, 0, 2))
            x = x.permute(1, 0, 2)

            # xlmk, ho = self.rnn_lmk(xlmk.permute(1, 0, 2))
            # xlmk = xlmk.permute(1, 0, 2)
            # x = torch.cat([x, xlmk], dim=-1)

            # x = x - torch.mean(x, dim=1, keepdim=True)

            # x = x.permute(0, 2, 1)
            # # x = self.conv_module(x)
            # attn = self.elem_atten(x).permute(0, 2, 1)
            # x = x.permute(0, 2, 1)

            reg_token = self.reg_token
            x_t = torch.cat([reg_token, x], dim=1)
            # x_t = x_t.reshape(1, n+1, -1) + self.pos_embedding[:, :n+1].to(x.device)
            x_t = self.transformer(x_t.permute(1, 0, 2)).permute(1, 0, 2)

            x_t1 = x_t[:, 0]
            # x_t2 = torch.sum(x_t[:, 1:], dim=1)
            # x = torch.cat([x_t1[:, i] for i in range(self.tokens)], dim=-1)
            x = x_t1
            
            # # x_r, ho = self.rnn(x.permute(1, 0, 2))
            # x = torch.cat([x_t1, x_t2], dim=-1)
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
        
        # hard samples
        quantile = torch.quantile(losses, 0.75, interpolation='nearest')
        mask[losses > quantile] = 2.

        # # noisy samples
        # quantile = torch.quantile(losses, 0.98, interpolation='nearest')
        # mask[losses > quantile] = 0.5

        return mask
    
    def compute_loss(self, preds, labels):
        if self.loss_type == 'l2':
            loss_l2 = F.mse_loss(preds, labels, reduction='none')
            mask = self.hard_sample_mask(loss_l2, False)
            # mask = self.hard_sample_mask(loss_l2, True)
            loss = torch.mean(loss_l2 * mask)
        elif self.loss_type == 'l1':
            loss_l1 = torch.abs(preds - labels)
            mask = self.hard_sample_mask(loss_l1, False)
            loss = torch.mean(loss_l1 * mask)
        elif self.loss_type == 'pcc':
            loss = 1 - self.calculate_apcc(preds, labels)
        
        return loss

    # def compute_reg_loss(self):
        


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
        lr = self.lr_schedulers().get_last_lr()[0]
        print('cur_lr', lr)

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
        data, labels = batch
        self.test_vids.extend(data['vid'])
        preds = self.forward_model(data)
        result = {"val_preds": preds}
        return result
    
    def test_epoch_end(self, test_step_outputs):

        for name, params in self.named_parameters():
            print(name, torch.mean(torch.sum(params**2)**0.5).item())

        preds = torch.cat([data['val_preds'] for data in test_step_outputs], dim=0)
        values = preds.detach().cpu().numpy()

        df = pd.DataFrame(values, columns=self.exp_names, index=self.test_vids)
        df.to_csv('dataset/abaw5_results/test.csv')

        return


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
