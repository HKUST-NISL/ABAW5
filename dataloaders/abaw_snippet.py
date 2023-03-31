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
import pickle

def create_transform(in_size=224):
    transform = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    return transform   

class Collator(object):
    def __init__(self, flag=False):
        super().__init__()
        self.flag = flag

    def __call__(self, data):
        '''
        Select a specific number of images randomly for the time being
        :param data:
        :return: batch_x torch.tensor{'images': bs, imgRandomLen, 299, 299, 3; 'age': bs; 'country': bs},
        batch_y torch.tensor: bs, 7;
        '''
        batch_x = {}

        batch_x['images'] = [x['images'] for x in data]
        batch_x['audio'] = [x['audio'] for x in data]
        batch_x['age'] = torch.stack([x['age'] for x in data])
        batch_x['country'] = torch.stack([x['country'] for x in data])
        # batch_x['age_con'] = torch.stack([x['age_con'] for x in data])
        batch_x['vid'] = [x['vid'] for x in data]
        # batch_x['intensity'] = np.stack([x['intensity'] for x in data])
        batch_y = torch.stack([x['intensity'] for x in data])
        batch_x['au_c'] = [x['au_c'] for x in data]
        batch_x['au_r'] = [x['au_r'] for x in data]

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
        indexList = ['Train', 'Val', 'Test']
        self.set_type = indexList[trainIndex]
        self.set_dir = self.set_type.lower()
        data_path = os.path.join(dataset_folder_path, self.set_dir, 'aligned')

        data_info_path = os.path.join(dataset_folder_path, 'data_info.csv')
        df = pd.read_csv(data_info_path)

        df_data = df[df['Split']==self.set_type]
        # print(df_data)

        self.snippet_size = args['snippet_size']
        self.input_size = args['input_size']
        self.sample_times = args['sample_times']
        self.features = args['features']
        self.audio_features = args['audio_features']
        self.feat_dir = self.data_dir
        # self.diff_dir = 'abaw5_diffs0' if args['diff_dir']=='' else args['diff_dir']
        # self.diff_dir = 'pipnet_diffs' if args['diff_dir']=='' else args['diff_dir']
        self.diff_dir = 'abaw5_diffs_rm' if args['diff_dir']=='' else args['diff_dir']

        self.transform = create_transform(self.input_size)
        self.all_image_lists = []
        self.video_dict = {}
        self.vid_list = []
        print('Initializing %s' % (self.set_dir))

        pickle_path = os.path.join(self.data_dir, self.set_type + '_data.pkl')
        if os.path.exists(pickle_path):

            with open(pickle_path, 'rb') as handle:
                data = pickle.load(handle)

            self.vid_list = data['vid_list']
            self.video_dict = data['video_dict']

        else:

            num_path = './dataset/%s_num.csv' % self.set_dir
            vids = []
            snums = []
            labels = []
            for file_id in tqdm(df_data['File_ID'].values):
            # for file_id in df_data['File_ID'].values[:100]:
                # file_id = os.path.basename(data_file)
                # loc = df['File_ID'] == '['+file_id+']'

                file_name = file_id.replace('[', '').replace(']', '')
                loc = df['File_ID'] == file_id
                info = df[loc]
                if info.empty: continue
                # assert info.iat[0, 1].lower() == indexList[trainIndex]
                data_entry = {}
                intensity = info.iloc[0, 2:9].tolist()
                age = info.iloc[0, 9]
                country = info.iloc[0, 10]
                assert country == 'United States' or 'South Africa'

                # data_entry['videoPath'] = data_file
                data_entry['intensity'] = np.array(intensity)

                feat_path = os.path.join(self.feat_dir , 'features', self.features, self.set_dir, file_name+'.npy')
                audio_feat_path = os.path.join(self.feat_dir , 'features', self.audio_features, self.set_dir, file_name+'.npy')

                data_entry['feat_path'] = feat_path
                data_entry['audio_feat_path'] = audio_feat_path
                data_entry['age'] = np.array(age)
                data_entry['country'] = np.array(0 if country == 'United States' else 1)

                au_info_path = os.path.join(self.data_dir, 'openface_align', self.set_dir, 
                                                    'aligned', file_name, file_name+'.csv')
                au_info = pd.read_csv(au_info_path).values
                au_info = au_info[:, 679:].copy()
                au_info_r = au_info[:, :17]
                au_info_c = au_info[:, 17:]

                data_entry['au_c'] = au_info_c
                data_entry['au_r'] = au_info_r

                self.video_dict[file_name] = data_entry
                self.vid_list.append(file_name)

                labels.append(data_entry['intensity'].reshape((1, -1)))

            data = {}
            data['vid_list'] = self.vid_list
            data['video_dict'] = self.video_dict

            with open(pickle_path, 'wb') as handle:
                pickle.dump(data, handle)

            df = pd.DataFrame(snums, columns=['numbers'], index=vids)
            df.to_csv(num_path)

        self.args = args

        self.vid_list = self.vid_list * self.sample_times
        # if trainIndex == 0:
            # self.vid_list = self.vid_list * self.sample_times
        self.data_total_length = len(self.vid_list)

        print('%s: videos: %d' % (indexList[trainIndex], len(self.vid_list)))

    def __getitem__(self, index):
        data = {}
        vid_name = self.vid_list[index]
        feat_path = self.video_dict[vid_name]['feat_path']
        audio_feat_path = self.video_dict[vid_name]['audio_feat_path']

        video_entry = self.video_dict[vid_name]

        feat_array = np.load(feat_path)

        # audio_feat_path = audio_feat_path.replace('mfcc', 'mfcc_align')
        audio_feat_path = audio_feat_path.replace('mfcc', 'mfcc_2')
        aud_array = np.load(audio_feat_path)

        data['images'] = torch.from_numpy(feat_array).float()
        data['audio'] = torch.from_numpy(aud_array).float()
        data['vid'] = vid_name
        intensity = torch.from_numpy(video_entry['intensity']).float()
        data['intensity'] = intensity
        data['age'] = torch.from_numpy(video_entry['age'])
        data['country'] = torch.from_numpy(video_entry['country'])

        data['au_c'] = torch.from_numpy(video_entry['au_c']).float()
        data['au_r'] = torch.from_numpy(video_entry['au_r']).float()

        return data

    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]


class ABAWDataModuleSnippet(pl.LightningDataModule):
    def __init__(self, **args):
        super().__init__()
        num_workers = args['num_workers']
        is_train = (args['train'] == 'True')
        flag = args['snippet_size'] == 0
        collate_fn = Collator(flag)

        train_set = ABAWDataset(0, **args)
        val_set = ABAWDataset(1, **args)
        test_set = ABAWDataset(2, **args)
        

        self.train_loader = DataLoader(dataset=train_set,
                                    batch_size=args['batch_size'],
                                    shuffle=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn)
        self.val_loader = DataLoader(dataset=val_set,
                                    batch_size=args['batch_size'],
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn)

        self.test_loader = DataLoader(dataset=test_set,
                                    batch_size=args['batch_size'],
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


if __name__ == '__main__':
  
    dataset = ABAWDataModuleSnippet(data_dir="./dataset/abaw5",
                             batch_size=32,
                             input_size=224,
                             snippet_size = 0,
                             sample_times=1,
                             num_workers=8,
                             features='res18_aff',
                             audio_features='mfcc',
                             train='True',
                             feat_dir='',
                             diff_dir=''
                             )

    for batch in tqdm(dataset.train_loader):
        pass
    for batch in tqdm(dataset.val_loader):
        pass
    for batch in tqdm(dataset.test_loader):
        pass


