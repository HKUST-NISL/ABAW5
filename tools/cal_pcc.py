import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
pd.set_option('display.precision', 3)
np.set_printoptions(precision=3)

cols = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']
csv_path = 'dataset/abaw5/data_info.csv'
df = pd.read_csv(csv_path)

df_train = df[df['Split'] == 'Train']
corrM_train = df_train[cols].corr()
# print('Train:')
# print(corrM_train)

df_val = df[df['Split'] == 'Val']
df_val_gt = df_val[cols].sort_index()


csv_path = 'dataset/abaw5_results/val_193.csv'
df_val_pred = pd.read_csv(csv_path, index_col=0).sort_index()


for vid in df_val_gt.index:
    if vid not in df_val_pred.index:
        print(vid)

        df_val_pred.loc[vid] = df_val_gt.mean()

df_val_pred = df_val_pred.sort_index()

preds = df_val_pred.values
labels = df_val_gt.values

def cal_pcc(preds, labels):
    preds_mean = np.mean(preds, axis=0, keepdims=True)
    labels_mean = np.mean(labels, axis=0, keepdims=True)

    pcc = np.sum((preds-preds_mean) * (labels-labels_mean), axis=0) / \
        np.clip((np.sum((preds-preds_mean)**2, axis=0) * np.sum((labels-labels_mean)**2, axis=0))**0.5, a_min=1e-8, a_max=None)

    return np.mean(pcc)


print(cal_pcc(preds, labels))

