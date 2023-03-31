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
        y, sr = librosa.load(input_path, sr=None)
        # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
        audio_mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, n_fft=512, hop_length=240, n_mels=40, pad_mode='reflect', htk=True)

        a = audio_mfccs.reshape(-1,)
        if len(a) <= 40960:
            b = np.append(a, a)
            c = np.append(b, b)
            c = np.append(c, c)
            audio_mfcc = c[8 * 1024: 40 * 1024]
        else:
            audio_mfcc = a[8 * 1024: 40 * 1024]
        aud = audio_mfcc.reshape(1024, 32)
        aud = aud.transpose(1, 0)
        np.save(output_path, aud)

    print("finish ALL", dt)

  