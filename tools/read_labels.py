import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
pd.set_option('display.precision', 3)
np.set_printoptions(precision=3)

cols = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']
csv_path = 'dataset/abaw5_results/predictions.csv'
df1 = pd.read_csv(csv_path, index_col=False)
print(df1)

csv_path = 'dataset/abaw5_results/predictions_1.csv'
df2 = pd.read_csv(csv_path, index_col=False)

df1_val = df1[cols]
df2_val = df2[cols]

df_m_vals = (df1_val + df2_val) * 0.5

df_m_vals['File_ID'] = df1['File_ID']
df_m_vals = df_m_vals[['File_ID'] + cols]
print(df_m_vals)

df_m_vals.to_csv('dataset/abaw5_results/predictions_m.csv', index=False)




# # csv_path = 'dataset/abaw5/data_info.csv'
# # df = pd.read_csv(csv_path)

# # df_train = df[df['Split'] == 'Train']
# # corrM_train = df_train[cols].corr()
# # # print('Train:')
# # # print(corrM_train)

# # df_val = df[df['Split'] == 'Val']
# # corrM_val = df_val[cols].corr()
# # # print('Val')
# # # print(corrM_val)

# # pca = PCA(n_components = 7)
  
# # X_train = pca.fit_transform(df_train[cols])
# # # X_test = pca.transform(X_test)
# # explained_variance = pca.explained_variance_ratio_
# # print('Train:')
# # print(explained_variance)
# # print(np.cumsum(explained_variance))

# # x = np.arange(7) + 1
# # plt.xlabel('Componets')
# # plt.ylabel('cumulative ratio')
# # plt.title('Train Set')
# # plt.plot(x, np.cumsum(explained_variance))
# # plt.savefig('./dataset/pca.png')


# X_train_pca = pca.transform(X_train)
# X_train_pca2 = (X_train - pca.mean_).dot(pca.components_.T)

# print(pca.mean_)
# print(pca.components_.T[:, :6])

# print((X_train_pca - X_train_pca2).mean())
# print(X_train_pca2.shape)




# pca = PCA(n_components = 7)
# X_train = pca.fit_transform(df_val[cols])
# # X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_
# print('Val:')
# print(explained_variance)
# print(np.cumsum(explained_variance))


