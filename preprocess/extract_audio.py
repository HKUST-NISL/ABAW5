import torchaudio
import torch
from tqdm import tqdm
import os
import numpy as np
import natsort
import glob


def processAudio(audioPath, saving_dir, subfolder, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'wav2vec':
        bundle = torchaudio.pipelines.WAV2VEC2_BASE # WAV2VEC2_BASE, HUBERT_BASE
    elif model_name == 'hubert':
        bundle = torchaudio.pipelines.HUBERT_BASE
    else:
        print('model name not wav2vec or hubert')
        quit()
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

if __name__ == '__main__':
    processAudio(audioPath='/data/abaw5/data/train/wav/',
                 saving_dir='/data/abaw5/hubert_features/',
                 subfolder='train/', model_name='hubert')

    processAudio(audioPath='/data/abaw5/data/val/wav/',
                 saving_dir='/data/abaw5/hubert_features/',
                 subfolder='val/', model_name='hubert')
    processAudio(audioPath='/data/abaw5/data/val/wav/',
                 saving_dir='/data/abaw5/wav2vec/',
                 subfolder='val/', model_name='wav2vec')

    processAudio(audioPath='/data/abaw5/data/test/wav/',
                 saving_dir='/data/abaw5/hubert_features/',
                 subfolder='test/', model_name='hubert')
    processAudio(audioPath='/data/abaw5/data/test/wav/',
                 saving_dir='/data/abaw5/wav2vec/',
                 subfolder='test/', model_name='wav2vec')