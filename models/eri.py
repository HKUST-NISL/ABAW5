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
from models.trans import Transformer, sinusoidal_embedding


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
        self.mode = args['mode']


        self.pretrained_path = args['pretrained']

        hidden_ch = 256
        self.tokens = 1

        dense_feat_ch = 0
        if self.mode == 'vamm' or self.mode == 'video':

            if 'res18' in self.features:
                feat_ch = 512
            elif 'resnet50' in self.features:
                feat_ch = 2048

            # with au
            feat_ch += 18 + 17
            # feat_ch += 32
            self.rnn = nn.GRU(feat_ch, hidden_ch, 2, batch_first=True)

            # self.pos_embedding = nn.Parameter(torch.zeros(1, self.snippet_size + self.tokens, hidden_ch))
            # self.pos_embedding = sinusoidal_embedding(1000, hidden_ch)
            self.reg_token = nn.Parameter(torch.randn(1, self.tokens, hidden_ch))

            n_head = 4
            n_layers = 4
            d_feed = 256
            self.transformer = Transformer(hidden_ch, n_layers, n_head, dim_head=64, mlp_dim=d_feed, dropout=0.2)

            dense_feat_ch += hidden_ch


        if self.mode == 'vamm' or self.mode == 'audio':
            self.rnn_aud = nn.GRU(1024, hidden_ch, 2, batch_first=True)
            self.reg_token_aud = nn.Parameter(torch.randn(1, self.tokens, hidden_ch))

            n_head = 4
            n_layers = 4
            d_feed = 256
            self.transformer_aud = Transformer(hidden_ch, n_layers, n_head, dim_head=64, mlp_dim=d_feed, dropout=0.2)
            dense_feat_ch += hidden_ch


        # self.head = nn.Linear(feat_ch, 7)
        self.head = nn.Sequential(
            nn.Linear(dense_feat_ch, 256, bias=False),
            # nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7, bias=False),
        )

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
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.5)
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


    def forward_model_seq(self, data):
        input = data['images']
        AU = data['au_r']
        AUC = data['au_c']
        audio = data['audio']

        if self.mode == 'vamm' or self.mode == 'video':
            feats = []
            for i in range(len(input)):
                x = input[i].to(self.device)
                au = AU[i].to(self.device)
                auc = AUC[i].to(self.device)
                # aud = audio[i].to(self.device)

                x = torch.cat([x, au, auc], dim=1)
                n, c = x.shape
                x = x.reshape(1, n, -1)

                x, ho = self.rnn(x)
    
                reg_token = self.reg_token
                x_t = torch.cat([reg_token, x], dim=1)
                x_t = self.transformer(x_t)
                x = x_t[:, 0]
                
                feats.append(x)
            feats = torch.cat(feats, dim = 0)

        if self.mode == 'vamm' or self.mode == 'audio':
            audio_feats = []
            for i in range(len(input)):
                x = audio[i].to(self.device)

                n, c = x.shape
                x = x.reshape(1, n, -1)

                x, ho = self.rnn_aud(x)
    
                reg_token = self.reg_token_aud
                x_t = torch.cat([reg_token, x], dim=1)
                x_t = self.transformer_aud(x_t)
                x = x_t[:, 0]
                
                audio_feats.append(x)
            audio_feats = torch.cat(audio_feats, dim = 0)

        if self.mode == 'vamm':
            feats = torch.cat([feats, audio_feats], dim=1)
        if self.mode == 'audio':
            feats = audio_feats
        preds = self.head(feats)
        return preds

    def forward_model_audio(self, data):
        input = data['images']
        AU = data['au_r']
        AUC = data['au_c']
        audio = data['audio']

        audio_feats = []
        for i in range(len(input)):
            x = audio[i].to(self.device)

            n, c = x.shape
            n_s = n % 8

            x = x[n_s:].reshape(-1, 1024)

            n, c = x.shape
            x = x.reshape(1, n, -1)

            x, ho = self.rnn_aud(x)
 
            reg_token = self.reg_token_aud
            x_t = torch.cat([reg_token, x], dim=1)
            x_t = self.transformer_aud(x_t)
            x = x_t[:, 0]
            
            audio_feats.append(x)


        audio_feats = torch.cat(audio_feats, dim = 0)
    
        # feats = torch.cat([feats, audio_feats], dim=1)
        preds = self.head(audio_feats)

        return preds
    
    def forward_model(self, data):
        
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

    def compute_reg_loss(self):

        loss_w = 0.
        for param in self.all_params:
            loss_w += torch.sum(param**2)
        
        return loss_w
        

    def training_step(self, batch, batch_idx):
        # TODO: add logging for each step, also calculate epoch loss in training_epoch_end
        # loss = self._calculate_loss(batch, mode="train")
        data, labels = batch
        preds = self.forward_model(data)
        loss = self.compute_loss(preds, labels)
        
        # loss_2 = self.compute_reg_loss()
        # loss = loss_1 + loss_2 * 0.00001

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

        l2_w = []
        for params in self.all_params:
            l2_w.append((torch.sum(params**2)**0.5).item())
        
        self.log('mean_w_l2', np.mean(l2_w), on_epoch=True, prog_bar=True)

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
        ind_col = 'File_ID'
        # l2_w = []
        # for name, params in self.named_parameters():
        #     print(name, torch.mean(torch.sum(params**2)**0.5).item())
        #     l2_w.append(torch.mean(torch.sum(params**2)**0.5).item())
        # print(np.mean(l2_w))

        df_info = pd.read_csv('dataset/abaw5/data_info.csv', index_col=0)

        df_gt = df_info[df_info['Split'] == 'Test']
        df_gt = df_gt[self.exp_names].sort_index()

        preds = torch.cat([data['val_preds'] for data in test_step_outputs], dim=0)
        values = preds.detach().cpu().numpy()

        df_test = pd.DataFrame(values, columns=self.exp_names)
        df_test[ind_col] = ['[' + x + ']' for x in self.test_vids]

        new_cols = ['File_ID'] + self.exp_names
        df_test = df_test[new_cols].set_index(ind_col)
        
        for vid in df_gt.index:
            if vid not in df_test.index:
                print(vid)
                df_test.loc[vid] = df_test.mean()
        
        df_test = df_test.sort_index()
        preds = df_test.values
        labels = df_gt.values

        # def cal_pcc(preds, labels):
        #     preds_mean = np.mean(preds, axis=0, keepdims=True)
        #     labels_mean = np.mean(labels, axis=0, keepdims=True)

        #     pcc = np.sum((preds-preds_mean) * (labels-labels_mean), axis=0) / \
        #         np.clip((np.sum((preds-preds_mean)**2, axis=0) * np.sum((labels-labels_mean)**2, axis=0))**0.5, a_min=1e-8, a_max=None)

        #     return np.mean(pcc)
        # print(cal_pcc(preds, labels))

        df_test.reset_index().to_csv('dataset/abaw5_results/predictions.csv', index=False)

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
