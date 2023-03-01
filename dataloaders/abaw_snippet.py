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
        batch_x['images'] = torch.stack([x['images'] for x in data])
        batch_x['age'] = torch.stack([x['age'] for x in data])
        batch_x['country'] = torch.stack([x['country'] for x in data])
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
        #self.imgRandomLen = 10 #for the time being
        self.sampling = SamplingStrategy()
        dataset_folder_path = args['dataset_folder_path']
        indexList = ['train', 'val', 'test']
        data_path = os.path.join(dataset_folder_path, indexList[trainIndex], 'aligned')

        data_info_path = os.path.join(dataset_folder_path, 'data_info.csv')
        df = pd.read_csv(data_info_path)

        self.snippet_size = args['snippet_size']
        self.input_size = args['input_size']

        self.transform = create_transform(self.input_size)
        self.all_image_lists = []
        self.video_dict = {}
        self.vid_list = []
        print('Initializing %s' % (indexList[trainIndex]))
        for data_file in glob.glob(data_path + '/*'):#[:100]:
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
            # get the indices
            image_paths = natsort.natsorted(glob.glob(data_file + '/' + folder + '_aligned/frame*.jpg'))
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
        self.data_total_length = len(self.vid_list)

        print('%s: videos: %d images: %d' % (indexList[trainIndex], len(self.vid_list), len(self.all_image_lists)))

    def __getitem__(self, index):
        data = {}
        # image_entry = self.all_image_lists[index]
        # image_path = image_entry['path']
        vid_name = self.vid_list[index]
        image_paths = self.video_dict[vid_name]['image_paths']

        video_entry = self.video_dict[vid_name]
        sel_paths = np.random.choice(image_paths, self.snippet_size, replace=False)
        inputs = []
        for path in sel_paths:
            input = self.transform(Image.open(path)).unsqueeze(0)
            inputs.append(input)

        data['images'] = torch.cat(inputs, 0)
        
        data['vid'] = vid_name
        data['intensity'] = torch.from_numpy(video_entry['intensity']).float()
        data['age'] = torch.from_numpy(video_entry['age'])
        data['country'] = torch.from_numpy(video_entry['country'])
        return data

    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]


class ABAWDataModule(pl.LightningDataModule):
    def __init__(self, **args):
        super().__init__()
        train_set = ABAWDataset(0, **args)
        val_set = ABAWDataset(1, **args)
        test_set = ABAWDataset(1, **args)
        collate_fn = Collator()

        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=args['batch_size'],
                                       shuffle=True,
                                       num_workers=8,
                                       collate_fn=collate_fn)
        self.val_loader = DataLoader(dataset=val_set,
                                     batch_size=args['batch_size'],
                                     shuffle=False,
                                     num_workers=8,
                                     collate_fn=collate_fn)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=args['batch_size'],
                                      shuffle=False,
                                      num_workers=8,
                                      collate_fn=collate_fn)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


if __name__ == '__main__':
  
    dataset = ABAWDataModule(dataset_folder_path="./dataset/abaw5",
                             batch_size=2,
                             input_size=224,
                             snippet_size = 30
                             )

    for batch in tqdm(dataset.train_loader):
        pass
    for batch in tqdm(dataset.val_loader):
        pass
    for batch in tqdm(dataset.test_loader):
        pass


