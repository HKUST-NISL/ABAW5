from distutils.util import strtobool
import torch
import os
from PIL import Image
from glob import glob
import numpy as np
from torchvision import transforms
import argparse
import shutil
from tqdm import tqdm
import cv2
import pandas as pd

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Emotion Training')
# Datasets
parser.add_argument('--data_dir',  default='/data/abaw5', type=str)
parser.add_argument('--out_dir',  default='/data/abaw5_diffs', type=str)


def run(args):
    w_size = 10
    cols = [str(i+1) for i in range(w_size)]
    # types = ['train', 'val']
    types = ['val', 'train']
    out_dir = args.out_dir
    for set_type in types:
        print(set_type)
        diff_dir = os.path.join('/data', 'abaw5_diffs', set_type)

        os.makedirs(diff_dir, exist_ok=True)

        vid_dirs = sorted(glob(os.path.join(args.data_dir, set_type, 'aligned', '*')))
        for vid_dir in tqdm(vid_dirs):

            vid_name = os.path.basename(vid_dir)
            if not vid_name.isdigit(): continue

            diff_path = os.path.join(diff_dir, vid_name+'.csv')
            img_paths = sorted(glob(os.path.join(vid_dir, vid_name + '_aligned', "*.jpg")))

            imgs = []
            for path in img_paths:
                imgs.append(cv2.imread(path))
            names = []
            diffs = []
            
            for i in range(len(img_paths))[w_size:]:
                diff_row = []
                names.append(os.path.basename(img_paths[i]))
                for j in range(w_size):
                    diff_row.append(np.mean(np.abs(imgs[i].astype(np.float32) - imgs[i-w_size+j].astype(np.float32))))
                
                diffs.append(diff_row)

            df = pd.DataFrame(data=diffs, index=names, columns=cols)

            df.to_csv(diff_path)


            
if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
