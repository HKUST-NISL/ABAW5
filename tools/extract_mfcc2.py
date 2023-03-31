import os
import sys
import tqdm
import numpy as np
import librosa
from matplotlib import pyplot as plt
  
root = "./dataset/abaw5/"


data_types = ['train', 'val', 'test']
dir_name = 'mfcc_2'

for dt in data_types:
    print(dt)

    wav_files = os.listdir(os.path.join(root, "raw", dt, "wav"))
    wav_files = sorted(wav_files)

    save_path  = os.path.join(root, "features", dir_name, dt)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for wav in tqdm.tqdm(wav_files, desc="extracting mfcc features") :
        input_path = os.path.join(root, "raw", dt, "wav", wav)
        output_path = os.path.join(save_path, wav.replace(".wav", ".npy"))

        feat_path = output_path.replace(dir_name, 'res18_aff')
        feat_array = np.load(feat_path)
        nv = feat_array.shape[0]

        y, sr = librosa.load(input_path, sr=None)
        # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
        # hop_l = int(np.ceil(len(y) / nv))
        audio_mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, n_fft=512, hop_length=240, n_mels=40, pad_mode='reflect', htk=True)

        a = audio_mfccs.reshape(-1,)
        n = a.shape[0]
        feat_ch = 1024
        while len(a) <= feat_ch * 40:
            a = np.append(a, a)
        aud = a[8*feat_ch:40*feat_ch]#.reshape(-1, feat_ch)
        aud = aud.reshape(1024, -1)
        aud = aud.transpose(1, 0)

        # print(audio_mfccs.shape)
        # aud = audio_mfccs[8:].transpose(1, 0).reshape(-1)

        # n = aud.shape[0]
        # feat_ch = 1024
        # aud = aud[:(n//feat_ch) * feat_ch]#.reshape(-1, feat_ch)

        # # aud = aud.reshape(1024, -1)
        # # aud = aud.transpose(1, 0)
        # aud = aud.reshape(-1, feat_ch)
        np.save(output_path, aud)

    print("finish ALL", dt)

  