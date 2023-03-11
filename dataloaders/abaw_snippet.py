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
        batch_x['mask'] = torch.stack([x['mask'] for x in data])
        batch_x['age'] = torch.stack([x['age'] for x in data])
        batch_x['country'] = torch.stack([x['country'] for x in data])
        batch_x['age_con'] = torch.stack([x['age_con'] for x in data])
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
        # self.sampling = SamplingStrategy()
        dataset_folder_path = args['data_dir']
        self.data_dir = dataset_folder_path
        indexList = ['train', 'val', 'test']
        self.set_type = indexList[trainIndex]
        data_path = os.path.join(dataset_folder_path, indexList[trainIndex], 'aligned')

        data_info_path = os.path.join(dataset_folder_path, 'data_info.csv')
        df = pd.read_csv(data_info_path)

        self.snippet_size = args['snippet_size']
        self.input_size = args['input_size']
        self.sample_times = args['sample_times']
        self.features = args['features']
        self.feat_dir = self.data_dir if args['feat_dir']=='' else args['feat_dir']
        self.diff_dir = 'abaw5_diffs0' if args['diff_dir']=='' else args['diff_dir']

        self.transform = create_transform(self.input_size)
        self.all_image_lists = []
        self.video_dict = {}
        self.vid_list = []
        print('Initializing %s' % (indexList[trainIndex]))

        nums = []
        labels = []
        for data_file in glob.glob(data_path + '/*'):
        # for data_file in glob.glob(data_path + '/*')[:1000]:
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
            df_path = os.path.join(self.data_dir, self.diff_dir, self.set_type, file_name+'.csv')
            diff_df = pd.read_csv(df_path, index_col=0)

            scores = diff_df['10'].values
            ind_orderd = np.argsort(scores).tolist()

            # while len(ind_orderd) < self.snippet_size:
            #     ind_orderd = ind_orderd + ind_orderd

            names = diff_df.index.to_list()
            img_names = [ names[ind] for ind in ind_orderd[:self.snippet_size]] 
            image_paths = sorted([os.path.join(data_file + '/' + folder + '_aligned', name) for name in img_names])
            
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

            nums.append(len(image_paths))
            labels.append(data_entry['intensity'].reshape((1, -1)))

        self.args = args

        self.vid_list = self.vid_list * self.sample_times
        # if trainIndex == 0:
            # self.vid_list = self.vid_list * self.sample_times
        self.data_total_length = len(self.vid_list)

        print('%s: videos: %d images: %d times: %d' % (
            indexList[trainIndex], len(self.vid_list)//self.sample_times, 
            len(self.all_image_lists), self.sample_times))

    def __getitem__(self, index):
        data = {}
        # image_entry = self.all_image_lists[index]
        # image_path = image_entry['path']
        vid_name = self.vid_list[index]
        image_paths = self.video_dict[vid_name]['image_paths']

        video_entry = self.video_dict[vid_name]

        # if self.snippet_size > 0:
        #     sel_paths = np.random.choice(image_paths, self.snippet_size, replace=False)
        # else:
        #     sel_paths = image_paths
        
        sel_paths = image_paths
        
        inputs = []

        for path in sel_paths:

            if self.features == 'image':
                input = self.transform(Image.open(path)).unsqueeze(0)
            else:
                img_name = os.path.basename(path)[:-4]
                feat_path = os.path.join(self.feat_dir , self.features+'_features', self.set_type, vid_name, img_name+'.npy')
                input = torch.from_numpy(np.load(feat_path)).unsqueeze(0)
            inputs.append(input)
        
        mask = torch.ones(self.snippet_size)
        if len(inputs) < self.snippet_size:
            mask[len(inputs):] = 0
            inputs.extend([torch.zeros(inputs[0].shape)] * (self.snippet_size - len(inputs)))

        mask = torch.matmul(mask.view(-1, 1), mask.view(1, -1))

        data['images'] = torch.cat(inputs, 0)
        data['vid'] = vid_name
        intensity = torch.from_numpy(video_entry['intensity']).float()
        data['intensity'] = intensity
        data['mask'] = mask.float()
        data['age'] = torch.from_numpy(video_entry['age'])
        data['country'] = torch.from_numpy(video_entry['country'])

        age = int(video_entry['age']) - 15
        if age > 34: age = 49
        if age < 0: age = 0
        age_bin = age // 5
        age_con = torch.zeros((8))
        age_con[age_bin] = 1
        age_con[7] = int(video_entry['country'])
        data['age_con'] = age_con
        
        return data

    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]


class ABAWDataModuleSnippet(pl.LightningDataModule):
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
  
    dataset = ABAWDataModuleSnippet(data_dir="./dataset/abaw5",
                             batch_size=2,
                             input_size=224,
                             snippet_size = 30,
                             sample_times=5,
                             )

    for batch in tqdm(dataset.train_loader):
        print(batch[0]['age_con'].shape)
    for batch in tqdm(dataset.val_loader):
        pass
    for batch in tqdm(dataset.test_loader):
        pass


