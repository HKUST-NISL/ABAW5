import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
pd.set_option('display.precision', 3)
np.set_printoptions(precision=3)


def calculate_apcc(preds, labels):
    preds_mean = np.mean(preds, axis=0, keepdims=True)
    labels_mean = np.mean(labels, axis=0, keepdims=True)

    pcc = np.sum((preds-preds_mean) * (labels-labels_mean), axis=0) / \
        np.clip((np.sum((preds-preds_mean)**2, axis=0) * np.sum((labels-labels_mean)**2, axis=0))**0.5, a_min=1e-8, a_max=None)

    return np.mean(pcc), pcc

cols = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']
csv_path = 'dataset/abaw5/data_info.csv'
df = pd.read_csv(csv_path)

df_train = df[df['Split'] == 'Train']
df_train_gt = df_train[cols].sort_index()

df_val = df[df['Split'] == 'Val']
df_val_gt = df_val[cols].sort_index()
# print(df_val_gt)

csv_path = 'dataset/abaw5_results/val_ref_107.csv'
df_val_pred = pd.read_csv(csv_path, index_col=0).sort_index()
# print(df_val_pred)

csv_path = 'dataset/abaw5_results/train_107.csv'
df_train_pred = pd.read_csv(csv_path, index_col=0).sort_index()

preds = df_train_pred.values
labels = df_train_gt.values
apcc, pcc0 = calculate_apcc(preds, labels)
# print('train apcc: %.3f,' % apcc, )

preds = df_val_pred.values
labels = df_val_gt.values
apcc, pcc1 = calculate_apcc(preds, labels)
# print('val apcc: %.3f' % apcc)


print('Train GT Corr:')
print(df_train_gt.corr())

print('Train Pred Corr:')
print(df_train_pred.corr())

print('Val GT Corr:')
print(df_val_gt.corr())

print('Val Pred Corr:')
print(df_val_pred.corr())

df_train_gt_usa = df_train[df_train['Country'] == 'United States'][cols]
df_train_gt_sa = df_train[df_train['Country'] == 'South Africa'][cols]

df_val_gt_usa = df_val[df_val['Country'] == 'United States'][cols]
df_val_gt_sa = df_val[df_val['Country'] == 'South Africa'][cols]


a = df_train_gt_usa.join(df_train_pred, rsuffix="2").values
apcc, pcc2 = calculate_apcc(a[:, :7], a[:, 7:])

a = df_train_gt_sa.join(df_train_pred, rsuffix="2").values
apcc, pcc3 = calculate_apcc(a[:, :7], a[:, 7:])

a = df_val_gt_usa.join(df_val_pred, rsuffix="2").values
apcc, pcc4 = calculate_apcc(a[:, :7], a[:, 7:])

a = df_val_gt_sa.join(df_val_pred, rsuffix="2").values
apcc, pcc5 = calculate_apcc(a[:, :7], a[:, 7:])


df_pcc = pd.DataFrame(np.vstack([pcc0, pcc1, pcc2, pcc3, pcc4, pcc5]), 
                      columns=cols, 
                      index=['train', 'val', 'train_usa', 'train_sa', 'val_usa', 'val_sa'])

df_pcc['apcc'] = df_pcc.mean(axis=1)

print(df_pcc)

df = (df_val_pred - df_val_pred.mean()) * (df_val_gt - df_val_gt.mean())
df_val_pred['pcc'] = df.mean(axis=1)

# print(df_val_pred.sort_values('pcc').head(10))



# print(df_val_gt.head(10))
# print(df_val_pred.head(10))
