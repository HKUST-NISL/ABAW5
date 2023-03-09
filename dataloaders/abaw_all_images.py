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
from dataloaders.abaw_snippet import create_transform

class ABAWDataset(Dataset):
    def __init__(self, trainIndex, **args):
        '''
        :param args: contains dataset_folder_path
        :param trainIndex: 0=train, 1=val, 2=test
        :returns country is 0 if US, 1 if SA
        '''
        self.input_image_size = args['input_size']
        self.transform = create_transform(self.input_image_size)
        dataset_folder_path = args['data_dir']
        indexList = ['train', 'val', 'test']
        data_path = os.path.join(dataset_folder_path, indexList[trainIndex], 'aligned')
        print('loading VGG features')
        self.data_path_feature = os.path.join(dataset_folder_path, indexList[trainIndex], 'vgg_features')

        data_info_path = os.path.join(dataset_folder_path, 'data_info.csv')
        df = pd.read_csv(data_info_path)

        self.all_image_lists = []
        self.video_dict = {}

        for data_file in glob.glob(data_path + '/*'):
            file_name = data_file.split('/')[-1]
            loc = df['File_ID'] == '['+file_name+']'
            info = df[loc]
            assert info.iat[0, 1].lower() == indexList[trainIndex]
            data_entry = {}
            intensity = info.iloc[0, 2:9].tolist()
            age = info.iloc[0, 9]
            country = info.iloc[0, 10]
            assert country == 'United States' or 'South Africa'

            data_entry['videoPath'] = data_file

            data_entry['intensity'] = np.array(intensity)
            folder = data_file.split('/')[-1]
            # get the indices
            image_paths = natsort.natsorted(glob.glob(data_file + '/' + folder + '_aligned/frame*.jpg'))
            data_entry['image_paths'] = image_paths
            data_entry['age'] = np.array(age)
            data_entry['country'] = np.array(0 if country == 'United States' else 1)

            self.video_dict[file_name] = data_entry

            for img_path in image_paths:
                this_image = {
                    'path': img_path,
                    'vid': file_name,
                }
                self.all_image_lists.append(this_image)

        self.args = args
        self.data_total_length = len(self.all_image_lists)

        print('Dataset size %s: %d' % (indexList[trainIndex], self.data_total_length))

    def __getitem__(self, index):
        data = {}
        image_entry = self.all_image_lists[index]
        image_path = image_entry['path']
        vid_name = image_entry['vid']
        video_entry = self.video_dict[vid_name]
        #image = cv2.imread(image_path)
        #image = cv2.resize(image, (self.input_image_size, self.input_image_size))
        #resized_image = image.transpose(2, 0, 1)
        data['vid'] = vid_name
        data['imagePath'] = image_path.split('/')[-1][:-4]
        if self.args['load_feature'] == 'True':
            featurePath = self.data_path_feature + '/' + vid_name + '/' + data['imagePath'] + '.npy'
            data['image'] = torch.from_numpy(np.load(featurePath))
        else:
            data['image'] = self.transform(Image.open(image_path))
        data['intensity'] = torch.from_numpy(video_entry['intensity']).float()
        data['age'] = torch.from_numpy(video_entry['age'])
        data['country'] = torch.from_numpy(video_entry['country'])
        return data

    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]


class ABAWDataModule_all_images(pl.LightningDataModule):
    def __init__(self, **args):
        super().__init__()
        train_set = ABAWDataset(0, **args)
        val_set = ABAWDataset(1, **args)
        test_set = ABAWDataset(1, **args)
        collate_fn = Collator()

        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=args['batch_size'],
                                       num_workers=10,
                                       collate_fn=collate_fn)
        self.val_loader = DataLoader(dataset=val_set,
                                     batch_size=args['batch_size'],
                                     num_workers=10,
                                     collate_fn=collate_fn)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=1,
                                      num_workers=10,
                                      collate_fn=collate_fn)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


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
        batch_x['image'] = torch.stack([x['image'] for x in data])
        batch_x['age'] = torch.stack([x['age'] for x in data])
        batch_x['country'] = torch.stack([x['country'] for x in data])
        batch_x['vid'] = [x['vid'] for x in data]
        batch_x['imagePath'] = [x['imagePath'] for x in data]
        batch_y = torch.stack([x['intensity'] for x in data])
        return batch_x, batch_y


if __name__ == '__main__':
    '''class ARGS(object):
        def __init__(self):
            self.dataset_folder_path = './dataset/abaw5/'
            self.input_image_size = 299
    args = ARGS()
    abaw = ABAWDataset(0, args)
    collate_fn = Collator()
    train_loader = DataLoader(dataset=abaw,
                              batch_size=2,
                              num_workers=0,
                              collate_fn=collate_fn,
                              shuffle=True)
    for batch in train_loader:
        print(batch)'''
    dataset = ABAWDataModule(dataset_folder_path="./dataset/",
                             batch_size=32,
                             input_image_size=299,
                             load_feature='True'
                             )
    # pbar = tqdm(len(dataset.train_loader))
    for batch in tqdm(dataset.val_loader):
        pass