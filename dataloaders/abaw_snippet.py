from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import glob
import random
from PIL import Image
import cv2
import natsort
from tqdm import tqdm
from dataloaders.sampling_strategy import SamplingStrategy
import torch
import os
from torchvision import transforms

def create_transform(in_size=224):
    transform = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    return transform   

class Collator(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        '''
        Select a specific number of images randomly for the time being
        :param data:
        :return: batch_x torch.tensor{'images': bs, imgRandomLen, 299, 299, 3; 'age': bs; 'country': bs},
        batch_y torch.tensor: bs, 7;
        '''
        batch_x = {}
        batch_x['images'] = [x['images'] for x in data] #torch.stack([x['images'] for x in data])
        batch_x['age'] = torch.stack([x['age'] for x in data])
        batch_x['country'] = torch.stack([x['country'] for x in data])
        batch_x['vid'] = torch.tensor([int(x['vid']) for x in data])
        # batch_x['intensity'] = np.stack([x['intensity'] for x in data])
        batch_y = torch.stack([x['intensity'] for x in data])
        return batch_x, batch_y


class ABAWDataset(Dataset):
    def __init__(self, trainIndex, **args):
        '''
        :param args: contains dataset_folder_path
        :param trainIndex: 0=train, 1=val, 2=test
        :returns country is 0 if US, 1 if SA
        '''
        dataset_folder_path = args['data_dir']
        indexList = ['train', 'val', 'test']
        data_path = os.path.join(dataset_folder_path, indexList[trainIndex], 'aligned')
        data_info_path = os.path.join(dataset_folder_path, 'data_info.csv')
        self.sampling_strategy = SamplingStrategy(os.path.join(dataset_folder_path, indexList[trainIndex], 'MaE_score'), sampling_choice=args['sampling_strategy'])
        df = pd.read_csv(data_info_path)

        if args['load_feature'] == 'smm':
            print('loading SMM features')
            self.data_path_feature = os.path.join(dataset_folder_path, indexList[trainIndex], 'features')
        elif args['load_feature'] == 'vgg':
            print('loading VGG features')
            self.data_path_feature = os.path.join(dataset_folder_path, indexList[trainIndex], 'vgg_features')

        self.snippet_size = args['snippet_size']
        self.input_size = args['input_size']
        self.sample_times = args['sample_times']

        self.transform = create_transform(self.input_size)
        self.all_image_lists = []
        self.video_dict = {}
        self.vid_list = []
        print('Initializing %s' % (indexList[trainIndex]))
        if args['load_feature'] == 'False':
            all_data_files = glob.glob(data_path + '/*')
        else:
            all_data_files = glob.glob(self.data_path_feature + '/*')
        for data_file in all_data_files[:2]:
            file_name = data_file.split('/')[-1]
            loc = df['File_ID'] == '['+file_name+']'
            info = df[loc]
            if info.empty: continue
            assert info.iat[0, 1].lower() == indexList[trainIndex]
            data_entry = {}
            intensity = info.iloc[0, 2:9].tolist()
            age = info.iloc[0, 9]
            country = info.iloc[0, 10]
            assert country == 'United States' or 'South Africa'

            # data_entry['videoPath'] = data_file

            data_entry['intensity'] = np.array(intensity)
            folder = data_file.split('/')[-1]
            if args['load_feature'] == 'False':
                image_paths = natsort.natsorted(glob.glob(data_file + '/' + folder + '_aligned/frame*.jpg'))
            else:
                image_paths = natsort.natsorted(glob.glob(self.data_path_feature + '/' + folder + '/*.npy'))
            data_entry['image_paths'] = image_paths
            data_entry['age'] = np.array(age)
            data_entry['country'] = np.array(0 if country == 'United States' else 1)

            self.video_dict[file_name] = data_entry
            self.vid_list.append(file_name)

            for img_path in image_paths:
                this_image = {
                    'path': img_path,
                    'vid': file_name,
                }
                self.all_image_lists.append(this_image)

        self.args = args
        self.vid_list = self.vid_list * self.sample_times
        # if trainIndex > 0:
        #     self.vid_list = self.vid_list * self.sample_times
        self.data_total_length = len(self.vid_list)

        print('%s: videos: %d images: %d' % (indexList[trainIndex], len(self.vid_list), len(self.all_image_lists)))

    def __getitem__(self, index):
        data = {}
        # image_entry = self.all_image_lists[index]
        # image_path = image_entry['path']
        vid_name = self.vid_list[index]
        image_paths = self.video_dict[vid_name]['image_paths']

        video_entry = self.video_dict[vid_name]
        sel_paths = self.sampling_strategy.get_sampled_paths(image_paths, self.snippet_size)
        inputs = []
        for path in sel_paths:
            if self.args['load_feature'] == 'False':
                input = self.transform(Image.open(path)).unsqueeze(0)
                inputs.append(input)
            else:
                featurePath = self.data_path_feature + '/' + vid_name + '/' + path.split('/')[-1][:-4] + '.npy'
                input = torch.from_numpy(np.load(featurePath)).unsqueeze(0)
                inputs.append(input)

        data['images'] = torch.cat(inputs, 0)
        data['vid'] = vid_name
        intensity = torch.from_numpy(video_entry['intensity']).float()
        # norm
        #intensity = (intensity - 0.3652) / 0.3592
        data['intensity'] = intensity
        age_min = 18.5
        age_max = 49
        age = torch.from_numpy(video_entry['age'])
        data['age'] = (age - age_min) / (age_max - age_min)
        data['country'] = torch.from_numpy(video_entry['country'])
        return data

    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]


class ABAWDataModule_snippet(pl.LightningDataModule):
    def __init__(self, **args):
        super().__init__()
        train_set = ABAWDataset(0, **args)
        val_set = ABAWDataset(1, **args)
        test_set = ABAWDataset(1, **args)
        collate_fn = Collator()

        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=args['batch_size'],
                                       shuffle=True,
                                       num_workers=args['num_workers'],
                                       collate_fn=collate_fn)
        self.val_loader = DataLoader(dataset=val_set,
                                     batch_size=args['batch_size'],
                                     shuffle=False,
                                     num_workers=args['num_workers'],
                                     collate_fn=collate_fn)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=args['batch_size'],
                                      shuffle=False,
                                      num_workers=args['num_workers'],
                                      collate_fn=collate_fn)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


if __name__ == '__main__':
    dataset = ABAWDataModule_snippet(data_dir="./dataset/",
                             batch_size=2,
                             input_size=299,
                             snippet_size=30,
                             sample_times=1,
                             sampling_strategy=1,
                             load_feature='smm'
                             )
    for batch in tqdm(dataset.val_loader):
        pass
    for batch in tqdm(dataset.train_loader):
        pass
    for batch in tqdm(dataset.test_loader):
        pass


