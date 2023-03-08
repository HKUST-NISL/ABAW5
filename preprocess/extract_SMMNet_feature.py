from models.smm_net import SMMNet
import torch
from dataloaders.abaw_all_images import ABAWDataModule_all_images
from tqdm import tqdm
import numpy as np
import os
from models.resnet import resnet50
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    dataset_path = "dataset/"
    #ckpt_path = '/data/pretrained/model-epoch=07-val_total=1.54.ckpt'
    #net = SMMNet().to(device)
    # ckpt = torch.load(ckpt_path)['state_dict']
    # net.load_state_dict(ckpt)

    net = resnet50(include_top=False, num_classes=8631)
    ckpt_path = 'pretrained/resnet50/resnet50_scratch_weight.pkl'
    with open(ckpt_path, 'rb') as f:
        obj = f.read()
    weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
    net.load_state_dict(weights)

    dataset = ABAWDataModule_all_images(data_dir=dataset_path,
                             batch_size=250,
                             input_size=224,
                             load_feature=False
                             )
    saving_dir = dataset_path + 'train/vgg_features/'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    for batch in tqdm(dataset.train_loader):
        image = batch[0]['image'].to(device)
        vid = batch[0]['vid']
        imagePath = batch[0]['imagePath']
        with torch.no_grad():
            features = net(image)[:,:,0,0]  # size: 64, 272

        for i in range(features.size()[0]):
            feature = features[i].cpu().detach().numpy()
            folder_dir = saving_dir + vid[i]
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            final_dir = folder_dir + '/' + imagePath[i]
            np.save(final_dir, feature)
        del image, features

    saving_dir = dataset_path + 'val/vgg_features/'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    for batch in tqdm(dataset.val_loader):
        image = batch[0]['image'].to(device)
        vid = batch[0]['vid']
        imagePath = batch[0]['imagePath']
        with torch.no_grad():
            features = net(image)[:,:,0,0]   # size: 64, 272

        for i in range(features.size()[0]):
            feature = features[i].cpu().detach().numpy()
            folder_dir = saving_dir + vid[i]
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            final_dir = folder_dir + '/' + imagePath[i]
            np.save(final_dir, feature)

