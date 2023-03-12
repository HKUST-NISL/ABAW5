import pandas as pd
pd.set_option('display.precision', 3)

cols = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']
csv_path = 'dataset/abaw5/data_info.csv'
df = pd.read_csv(csv_path)


df_train = df[df['Split'] == 'Train']
corrM_train = df_train[cols].corr()

print('Train:')
print(corrM_train)


df_val = df[df['Split'] == 'Val']
corrM_val = df_val[cols].corr()
print('Val')
print(corrM_val)
