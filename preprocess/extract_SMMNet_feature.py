from models.smm_net import SMMNet
import torch
from dataloaders.abaw_all_images import ABAWDataModule
from tqdm import tqdm
import numpy as np
import os

if __name__ == '__main__':
    dataset_path = "./dataset/"
    ckpt_path = ''
    net = SMMNet()
    ckpt = torch.load(ckpt_path)['state_dict']
    net.load_state_dict(ckpt)
    dataset = ABAWDataModule(dataset_folder_path=dataset_path,
                             batch_size=64,
                             input_image_size=299,
                             )
    saving_dir = dataset_path + 'train/features/'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    for batch in tqdm(dataset.train_loader):
        image = batch[0]['image']
        vid = batch[0]['vid']
        imagePath = batch[0]['imagePath']
        with torch.no_grad():
            features = net(image)  # size: 64, 272

        for i in range(features.size()[0]):
            feature = features[i].cpu().detach().numpy()
            folder_dir = saving_dir + vid[i]
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            final_dir = folder_dir + '/' + imagePath[i]
            np.save(final_dir, feature)

    saving_dir = dataset_path + 'val/features/'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    for batch in tqdm(dataset.val_loader):
        image = batch[0]['image']
        vid = batch[0]['vid']
        imagePath = batch[0]['imagePath']
        with torch.no_grad():
            features = net(image)  # size: 64, 272

        for i in range(features.size()[0]):
            feature = features[i].cpu().detach().numpy()
            folder_dir = saving_dir + vid[i]
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            final_dir = folder_dir + '/' + imagePath[i]
            np.save(final_dir, feature)
