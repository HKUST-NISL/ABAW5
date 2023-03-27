import os
import sys
import tqdm
import numpy as np
import librosa
from matplotlib import pyplot as plt
  
root = "./dataset/abaw5/"


data_types = ['train', 'val', 'test']

for dt in data_types:
    print(dt)

    wav_files = os.listdir(os.path.join(root, "raw", dt, "wav"))
    wav_files = sorted(wav_files)

    save_path  = os.path.join(root, "features/mfcc/", dt)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for wav in tqdm.tqdm(wav_files, desc="extracting mfcc features") :
        input_path = os.path.join(root, "raw", dt, "wav", wav)
        output_path = os.path.join(save_path, wav.replace(".wav", ".npy"))
        y, sr = librosa.load(input_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        np.save(output_path, mfccs)

    print("finish ALL", dt)

  