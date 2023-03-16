import math

from models.smm_net import SMMNet
import torch
from dataloaders.abaw_all_images import ABAWDataModule_all_images
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from models.resnet import resnet50
import pickle
import natsort
import glob
import cv2
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = "dataset/"
ckpt_path = 'pretrained/model-epoch=07-val_total=1.54.ckpt'
net = SMMNet().to(device)
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
net.load_state_dict(ckpt)
batch = 64

'''net = resnet50(include_top=False, num_classes=8631).to(device)
ckpt_path = 'pretrained/resnet50/resnet50_scratch_weight.pkl'
with open(ckpt_path, 'rb') as f:
    obj = f.read()
weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
net.load_state_dict(weights)'''

dataset = ABAWDataModule_all_images(data_dir=dataset_path, batch_size=2, input_size=299, load_feature='False')

def getListOfRealignedVideos(blackImageFile):
    df = pd.read_csv(blackImageFile)
    videoList = set(df.iloc[:, 0].tolist())
    return videoList

def create_transform(in_size=299):
    transform = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    return transform

def extract_train_all(AU_feature=True):
    if AU_feature:
        saving_dir = dataset_path + 'train/AU_features/'
    else:
        saving_dir = dataset_path + 'train/features/'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    for batch in tqdm(dataset.train_loader):
        image = batch[0]['image'].to(device)
        vid = batch[0]['vid']
        imagePath = batch[0]['imagePath']
        with torch.no_grad():
            features = net(image, outputAU=AU_feature) #[:, :, 0, 0]  # size: 64, 272

        for i in range(features.size()[0]):
            feature = features[i].cpu().detach().numpy()
            folder_dir = saving_dir + vid[i]
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            final_dir = folder_dir + '/' + imagePath[i]
            np.save(final_dir, feature)
        del image, features

def extract_val_all(AU_feature=True):
    if AU_feature:
        saving_dir = dataset_path + 'val/AU_features/'
    else:
        saving_dir = dataset_path + 'val/features/'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    for batch in tqdm(dataset.val_loader):
        image = batch[0]['image'].to(device)
        vid = batch[0]['vid']
        imagePath = batch[0]['imagePath']
        with torch.no_grad():
            features = net(image, outputAU=True) #[:, :, 0, 0]  # size: 64, 272

        for i in range(features.size()[0]):
            feature = features[i].cpu().detach().numpy()
            folder_dir = saving_dir + vid[i]
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            final_dir = folder_dir + '/' + imagePath[i]
            np.save(final_dir, feature)


transformSMM = create_transform(299)
def extract_realigned(choice):
    saving_dir = dataset_path + choice + '/realigned_features/'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    files = natsort.natsorted(glob.glob(dataset_path + '/' + choice + "/re_aligned/*"))
    for i in tqdm(range(len(files))):
        dir_sub = files[i]
        folder = dir_sub.split('/')[-1]
        image_path = dir_sub + '/' + folder + '_aligned'
        imageFiles = natsort.natsorted(glob.glob(image_path + "/frame*.jpg"))
        images = []
        for dir_sub_vid_img in imageFiles:
            image = transformSMM(Image.open(dir_sub_vid_img))
            images.append(image)
        images = torch.stack(images).to(device)
        features = []
        with torch.no_grad():
            for i in range(0, images.size()[0], batch):
                feature = net(images[i:(i+batch)])
                features.append(feature)
        features = torch.cat(features)
        assert features.size()[0] == images.size()[0]
        for i in range(features.size()[0]):
            feature = features[i].cpu().detach().numpy()
            folder = imageFiles[i].split('/')[-3]
            file = imageFiles[i].split('/')[-1][:-4]
            folder_dir = saving_dir + folder
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            final_dir = folder_dir + '/' + file
            np.save(final_dir, feature)

#extract_realigned('train')
#extract_realigned('val')
extract_train_all(True)
extract_val_all(True)


