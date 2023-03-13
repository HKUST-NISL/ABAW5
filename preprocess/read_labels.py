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
corrM_val = df_val[cols].corr()
# print('Val')
# print(corrM_val)

pca = PCA(n_components = 7)
  
X_train = pca.fit_transform(df_train[cols])
# X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print('Train:')
print(explained_variance)
print(np.cumsum(explained_variance))

x = np.arange(7) + 1
plt.xlabel('Componets')
plt.ylabel('cumulative ratio')
plt.title('Train Set')
plt.plot(x, np.cumsum(explained_variance))
plt.savefig('./dataset/pca.png')


pca = PCA(n_components = 7)
X_train = pca.fit_transform(df_val[cols])
# X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print('Val:')
print(explained_variance)
print(np.cumsum(explained_variance))


