import torchaudio
import torch
from tqdm import tqdm
import os
import numpy as np
import natsort
import glob

audioPath = '/Users/adia/Documents/HKUST/papers/abaw5/abaw/datasets/val/wav/'
saving_dir = 'dataset/wav2vec_features/'
subfolder = 'val/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bundle = torchaudio.pipelines.WAV2VEC2_BASE # WAV2VEC2_BASE, HUBERT_BASE
model = bundle.get_model().to(device)

if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
saving_dir = saving_dir+subfolder
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
files = natsort.natsorted(glob.glob(audioPath + "/*.wav"))
for i in tqdm(range(len(files))):
    dir_sub = files[i]
    folder = dir_sub.split('/')[-1][:-4]
    waveform, sample_rate = torchaudio.load(dir_sub)
    waveform = waveform.to(device)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    features, _ = model.extract_features(waveform)
    feature = features[-1][0].cpu().detach().numpy()
    np.save(saving_dir+folder, feature)