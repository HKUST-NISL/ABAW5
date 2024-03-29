import torch
from dataloaders.abaw_all_images import ABAWDataModule_all_images
from tqdm import tqdm
import numpy as np
import os
import sys
from models import *
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(sys.argv)
if __name__ == '__main__':
    dataset_path = "dataset/abaw5"
    if len(sys.argv) < 2 or sys.argv[1] == 'smm':
        ckpt_path = './pretrained/model-epoch=07-val_total=1.54.ckpt'
        net = SMMNet().to(device)
        ckpt = torch.load(ckpt_path)['state_dict']
        net.load_state_dict(ckpt)
        out_name = 'smm2_pip_features'
        in_size = 299
    elif sys.argv[1] == 'res50':
        net = resnet50(include_top=False).to(device)
        ckpt_path = 'pretrained/resnet50_ft_weight.pkl'
        with open(ckpt_path, 'rb') as f:
            obj = f.read()
        ckpt = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        ckpt_new = {}
        for key in ckpt.keys():
            if 'fc.' not in key:
                ckpt_new[key] = ckpt[key]
        net.load_state_dict(ckpt_new)
        out_name = sys.argv[1]+'_features'
        in_size = 224
    elif sys.argv[1] == 'effnetb0':
        net = effnetb0().to(device)
        ckpt_path = 'pretrained/state_vggface2_enet0_new.pt'
        ckpt = torch.load(ckpt_path)
        ckpt_new = {}
        for key in ckpt.keys():
            ckpt_new['model.'+key] = ckpt[key]
        net.load_state_dict(ckpt_new)
        out_name = sys.argv[1]+'_features'
        in_size = 224
    elif sys.argv[1] == 'effnetb0_aff':
        net = effnetb0().to(device)
        ckpt_path = 'pretrained/effnebb0_affect.pth'
        ckpt = torch.load(ckpt_path)
        ckpt_new = {}
        for key in ckpt.keys():
            if 'classifier' not in key:
                ckpt_new[key] = ckpt[key]
        net.load_state_dict(ckpt_new)
        out_name = sys.argv[1]+'_pip_features'
        in_size = 224
    elif sys.argv[1] == 'effnetb0_raf':
        net = effnetb0().to(device)
        ckpt_path = 'pretrained/effnebb0_rafdb.pth'
        ckpt = torch.load(ckpt_path)
        ckpt_new = {}
        for key in ckpt.keys():
            if 'classifier' not in key:
                ckpt_new[key] = ckpt[key]
        net.load_state_dict(ckpt_new)
        out_name = sys.argv[1]+'_features'
        in_size = 224
    elif sys.argv[1] == 'res18':
        net = ResNetEmo(version=18).to(device)
        ckpt_path = 'pretrained/res18_affect.pth'
        ckpt = torch.load(ckpt_path)
        ckpt_new = {}
        for key in ckpt.keys():
            if 'fc' not in key:
                ckpt_new[key] = ckpt[key]
        net.load_state_dict(ckpt_new)
        out_name = sys.argv[1]+'_features'
        in_size = 224
    print('load ok: %s' % ckpt_path)
    

    dataset = ABAWDataModule_all_images(data_dir=dataset_path,
                             batch_size=250,
                             input_size=in_size,
                             load_feature=False
                             )
    # saving_dir = os.path.join(dataset_path, out_name, 'train')
    # print(saving_dir)
    # if not os.path.exists(saving_dir):
    #     os.makedirs(saving_dir)
    # for batch in tqdm(dataset.train_loader):
    #     image = batch[0]['image'].to(device)
    #     vid = batch[0]['vid']
    #     imagePath = batch[0]['imagePath']
    #     with torch.no_grad():
    #         features = net(image).flatten(1)
    #     for i in range(features.size()[0]):
    #         feature = features[i].cpu().detach().numpy()
    #         folder_dir = os.path.join(saving_dir, vid[i])
    #         if not os.path.exists(folder_dir):
    #             os.mkdir(folder_dir)
    #         feat_path = os.path.join(folder_dir, imagePath[i])
    #         np.save(feat_path, feature)


    # saving_dir = os.path.join(dataset_path, out_name, 'val')
    # print(saving_dir)
    # if not os.path.exists(saving_dir):
    #     os.makedirs(saving_dir)
    # for batch in tqdm(dataset.val_loader):
    #     image = batch[0]['image'].to(device)
    #     vid = batch[0]['vid']
    #     imagePath = batch[0]['imagePath']
    #     with torch.no_grad():
    #         features = net(image).flatten(1)

    #     for i in range(features.size()[0]):
    #         feature = features[i].cpu().detach().numpy()
    #         folder_dir = os.path.join(saving_dir, vid[i])
    #         if not os.path.exists(folder_dir):
    #             os.mkdir(folder_dir)
    #         feat_path = os.path.join(folder_dir, imagePath[i])
    #         np.save(feat_path, feature)

    saving_dir = os.path.join(dataset_path, out_name, 'test')
    print(saving_dir)
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    for batch in tqdm(dataset.test_loader):
        image = batch[0]['image'].to(device)
        vid = batch[0]['vid']
        imagePath = batch[0]['imagePath']
        with torch.no_grad():
            features = net(image).flatten(1)

        for i in range(features.size()[0]):
            feature = features[i].cpu().detach().numpy()
            folder_dir = os.path.join(saving_dir, vid[i])
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            feat_path = os.path.join(folder_dir, imagePath[i])
            np.save(feat_path, feature)

    

    