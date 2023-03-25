import os

import tqdm

  
root = "./dataset/abaw5/"


data_types = ['train', 'val', 'test']

for dt in data_types:
    print(dt)

    wav_files = os.listdir(os.path.join(root, "raw", dt, "wav"))
    wav_files = sorted(wav_files)

    save_path  = os.path.join(root, "features/deepspectrum/", dt)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for wav in tqdm.tqdm(wav_files, desc="extracting deepspectrum features") :
        input_path = os.path.join(root, "raw", dt, "wav", wav)
        output_path = os.path.join(save_path, wav.replace(".wav",".csv"))
        cmd = f"deepspectrum features {input_path}  -nl -en densenet121 -fl avg_pool -m mel -o {output_path}"
        os.system(cmd)
        #print("finish: ", wav)

    print("finish ALL", dt)

  