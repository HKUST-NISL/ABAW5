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
        if not self.flag:
            batch_x['images'] = torch.stack([x['images'] for x in data])
            batch_x['mask'] = torch.stack([x['mask'] for x in data])
        else:
            batch_x['images'] = [x['images'] for x in data]
        batch_x['age'] = torch.stack([x['age'] for x in data])
        batch_x['country'] = torch.stack([x['country'] for x in data])
        batch_x['age_con'] = torch.stack([x['age_con'] for x in data])
        batch_x['vid'] = [x['vid'] for x in data]
        batch_x['au_c'] = [x['au_c'] for x in data]
        batch_x['au_r'] = [x['au_r'] for x in data]
        batch_x['gaze'] = [x['gaze'] for x in data]
        batch_x['pose'] = [x['pose'] for x in data]
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
        indexList = ['Train', 'Val', 'Test']
        self.set_type = indexList[trainIndex]
        self.set_dir = self.set_type.lower()
        self.align_path = os.path.join(dataset_folder_path, args['aligned'], self.set_dir, 'aligned')

        data_info_path = os.path.join(dataset_folder_path, 'data_info.csv')
        df = pd.read_csv(data_info_path)

        df_data = df[df['Split']==self.set_type]
        # print(df_data)

        self.snippet_size = args['snippet_size']
        self.input_size = args['input_size']
        self.sample_times = args['sample_times']
        self.features_path = os.path.join(dataset_folder_path, args['features'], self.set_dir)
        #self.feat_dir = self.data_dir if args['feat_dir']=='' else args['feat_dir']
        # self.diff_dir = 'abaw5_diffs0' if args['diff_dir']=='' else args['diff_dir']
        #self.diff_dir = 'pipnet_diffs' if args['diff_dir']=='' else args['diff_dir']
        # self.diff_dir = 'abaw5_diffs_rm' if args['diff_dir']=='' else args['diff_dir']
        #self.lmk_dir = 'pipnet_landmarks'


        self.transform = create_transform(self.input_size)
        self.all_image_lists = []
        self.video_dict = {}
        self.vid_list = []
        print('Initializing %s' % (self.set_dir))

        num_path = './dataset/%s_num.csv' % self.set_dir
        vids = []
        snums = []
        nums = []
        labels = []
        all_data_files = glob.glob(self.features_path + '/*')
        for data_file in all_data_files: #[:2]:
            file_name = data_file.split('/')[-1]
            file_id = '['+file_name+']'
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
            image_paths = natsort.natsorted(glob.glob(self.features_path + '/' + file_name + '/*.npy'))
            data_entry['image_paths'] = image_paths
            data_entry['age'] = np.array(age)
            data_entry['country'] = np.array(0 if country == 'United States' else 1)

            data_entry['au_csv'] = self.align_path + '/' + file_name + '/' + file_name + '.csv'

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

        df = pd.DataFrame(snums, columns=['numbers'], index=vids)
        df.to_csv(num_path)

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
        vid_name = self.vid_list[index]
        image_paths = self.video_dict[vid_name]['image_paths']
        video_entry = self.video_dict[vid_name]
        sel_paths = image_paths

        inputs = []

        for path in sel_paths:
            img_name = os.path.basename(path)[:-4]
            '''if self.features == 'image':
                input = self.transform(Image.open(path)).unsqueeze(0)
            else:'''
            feat_path = os.path.join(self.features_path, vid_name, img_name+'.npy')
            input = torch.from_numpy(np.load(feat_path)).unsqueeze(0)
            inputs.append(input)

        if self.snippet_size > 0:
            tokens = 1
            mask = torch.ones(self.snippet_size + tokens)
            if len(inputs) < self.snippet_size:
                mask[len(inputs) + tokens:] = 0
                inputs.extend([torch.zeros(inputs[0].shape)] * (self.snippet_size - len(inputs)))

            mask = torch.matmul(mask.view(-1, 1), mask.view(1, -1))
            data['mask'] = mask.float()

        data['images'] = torch.cat(inputs, 0)
        data['vid'] = vid_name
        intensity = torch.from_numpy(video_entry['intensity']).float()
        data['intensity'] = intensity
        data['age'] = torch.from_numpy(video_entry['age'])
        data['country'] = torch.from_numpy(video_entry['country'])

        info = pd.read_csv(video_entry['au_csv']).values
        au_info = info[:, 679:]
        au_info_r = au_info[:, :17]
        au_info_c = au_info[:, 17:]
        gaze = info[:, 5:13]
        pose = info[:, 296:299]
        data['au_r'] = au_info_r
        data['au_c'] = au_info_c
        data['gaze'] = gaze #size: 8
        data['pose'] = pose #size: 3

        data['au_c'] = torch.from_numpy(data['au_c']).float()
        data['au_r'] = torch.from_numpy(data['au_r']).float()
        data['gaze'] = torch.from_numpy(data['gaze']).float()
        data['pose'] = torch.from_numpy(data['pose']).float()

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
        num_workers = args['num_workers']
        train_set = ABAWDataset(0, **args)
        val_set = ABAWDataset(1, **args)
        test_set = ABAWDataset(2, **args)
        flag = args['snippet_size'] == 0
        collate_fn = Collator(flag)

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
                             batch_size=2,
                             input_size=224,
                             snippet_size = 30,
                             sample_times=5,
                             num_workers=8,
                             features='smm',
                             feat_dir='',
                             diff_dir=''
                             )

    for batch in tqdm(dataset.train_loader):
        pass
    for batch in tqdm(dataset.val_loader):
        pass
    for batch in tqdm(dataset.test_loader):
        pass


