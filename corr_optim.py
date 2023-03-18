import torch
from torch import nn
import pandas as pd
import numpy as np
import torch.optim as optim


class CorrOptim(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.diff = nn.Parameter(torch.zeros(n, 7))

        self.target = torch.tensor([[ 1.   ,  0.251, -0.382, -0.312, -0.272, -0.457, -0.289],
                        [ 0.251,  1.   , -0.514, -0.374, -0.43 , -0.606, -0.081],
                        [-0.382, -0.514,  1.   ,  0.143,  0.268,  0.732,  0.328],
                        [-0.312, -0.374,  0.143,  1.   ,  0.316,  0.154,  0.15 ],
                        [-0.272, -0.43 ,  0.268,  0.316,  1.   ,  0.204,  0.092],
                        [-0.457, -0.606,  0.732,  0.154,  0.204,  1.   ,  0.27 ],
                        [-0.289, -0.081,  0.328,  0.15 ,  0.092,  0.27 ,  1.   ]])

    
    def forward(self, x):

        x = x + self.diff
        corr = torch.corrcoef(x.T)

        loss = nn.functional.mse_loss(corr, self.target) + 20 * torch.mean(self.diff**2)

        return loss
    
if __name__ == '__main__':
    csv_path = 'dataset/abaw5_results/val_107.csv'
    df_val_pred = pd.read_csv(csv_path, index_col=0).sort_index()

    x = torch.from_numpy(df_val_pred.values).float()
    print(x.shape)
    net = CorrOptim(n = 4657)

    y = net(x)

    epochs = 10000
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for i in range(epochs):

        loss = net(x)
        print(i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

    print(x + net.diff)

    x = x + net.diff
    corr = torch.corrcoef(x.T)

    print(corr)


    x = x.detach().cpu().numpy()
    cols = df_val_pred.columns
    indx = df_val_pred.index

    df_preds = pd.DataFrame(x, columns=cols, index=indx)
    df_preds.to_csv('dataset/abaw5_results/val_ref_107.csv')



    