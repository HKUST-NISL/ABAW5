import os
import sys
import tqdm
import numpy as np
import librosa
from matplotlib import pyplot as plt
  
root = "./dataset/abaw5/"


data_types = ['train', 'val', 'test']
dir_name = 'mfcc_align'

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
        
        hop_l = int(np.ceil(len(y) / nv))
        audio_mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024, hop_length=hop_l, pad_mode='reflect', htk=True)

        na = audio_mfccs.shape[1]

        if na != nv:
            if na < nv:
                audio_mfccs = np.concatenate([audio_mfccs, audio_mfccs[:, na-nv:]], axis=-1)
            else:
                audio_mfccs = audio_mfccs[:, :nv]

        # audio_mfccs = audio_mfccs.reshape(-1, 40)
        aud = audio_mfccs[8:].transpose(1, 0)
        np.save(output_path, aud)

    print("finish ALL", dt)

  