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
parser.add_argument('--data_dir',  default='./dataset/abaw5/openface_align', type=str)
parser.add_argument('--out_dir',  default='./dataset/abaw5/abaw5_diffs_rm', type=str)


def run(args):
    w_size = 10
    cols = [str(i+1) for i in range(w_size)]
    # types = ['train', 'val']
    # types = ['val', 'train']
    types = ['test']
    out_dir = args.out_dir
    for set_type in types:
        print(set_type)
        diff_dir = os.path.join(args.out_dir, set_type)
        frame_count = 0

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
            
            for i in range(len(img_paths)):
                diff_row = []
                if np.sum(imgs[i]) == 0: continue
                names.append(os.path.basename(img_paths[i]))
                for j in range(w_size):
                    i_pre = i - (j+1)
                    if i_pre < 0:
                        diff = 0
                    else:
                        diff = np.mean(np.abs(imgs[i].astype(np.float32) - imgs[i_pre].astype(np.float32)))

                    diff_row.append(diff)
                
                diffs.append(diff_row)
                frame_count += 1

            df = pd.DataFrame(data=diffs, index=names, columns=cols)

            df.to_csv(diff_path)

        print(set_type, frame_count)


            
if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
