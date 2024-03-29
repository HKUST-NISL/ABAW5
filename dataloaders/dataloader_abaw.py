from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import glob
import random
from PIL import Image


class ABAWDataset(Dataset):
    def __init__(self, trainIndex, **args):
        '''
        :param args: contains dataset_folder_path
        :param trainIndex: 0=train, 1=val, 2=test
        :returns country is 0 if US, 1 if SA
        '''
        dataset_folder_path = args['dataset_folder_path']
        indexList = ['train', 'val', 'test']
        data_path = dataset_folder_path + indexList[trainIndex] + '/processed/'

        data_info_path = dataset_folder_path + 'data_info.csv'
        df = pd.read_csv(data_info_path)
        self.total_data = []

        input_image_size = args['input_image_size']

        for data_file in glob.glob(data_path + '/*.npy'):
            file_name = data_file.split('/')[-1][:-4]
            loc = df['File_ID'] == '['+file_name+']'
            info = df[loc]
            assert info.iat[0, 1].lower() == indexList[trainIndex]
            data_entry = {}
            intensity = info.iloc[0, 2:9].tolist()
            age = info.iloc[0, 9]
            country = info.iloc[0, 10]
            assert country == 'United States' or 'South Africa'

            images = np.load(data_file)
            height = images.shape[2]
            if height != input_image_size:
                resized_images = []
                for i in range(images.shape[0]):
                    image = images[i]
                    image = Image.fromarray(np.uint8(image))
                    image = image.resize((input_image_size, input_image_size))
                    resized_images.append(np.array(image))
                resized_images = np.stack(resized_images)
                images = resized_images

            data_entry['images'] = images
            data_entry['intensity'] = np.array(intensity)
            data_entry['age'] = np.array(age)
            data_entry['country'] = np.array(0 if country == 'United States' else 1)
            self.total_data.append(data_entry)

        self.args = args
        self.data_total_length = len(self.total_data)

        print(F'total_len = {self.data_total_length}')

    def __getitem__(self, index):
        return self.total_data[index]

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
                                       num_workers=2,
                                       collate_fn=collate_fn,
                                       shuffle=True)
        self.val_loader = DataLoader(dataset=val_set,
                                     batch_size=args['batch_size'],
                                     num_workers=2,
                                     collate_fn=collate_fn,
                                     shuffle=False)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=1,
                                      num_workers=2,
                                      collate_fn=collate_fn,
                                      shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class Collator(object):
    def __init__(self, imgRandomLen=10):
        super().__init__()
        self.imgRandomLen = imgRandomLen

    def __call__(self, data):
        '''
        Select a specfic number of images randomly for the time being
        :param data:
        :return: batch_x {'images': bs, imgRandomLen, 299, 299, 3; 'age': bs; 'country': bs},
        batch_y np.array: bs, 7;
        '''
        batch_x = {}
        images = [x['images'] for x in data]
        final_images = []
        for images_element in images:
            #random_indexes = random.sample(np.arange(images_element.shape[0]).tolist(), self.imgRandomLen)
            #random_indexes.sort()
            random_indexes = random.randint(0, images_element.shape[0]-self.imgRandomLen)
            random_images = images_element[random_indexes:(random_indexes+self.imgRandomLen)]
            final_images.append(random_images)
        batch_x['images'] = np.stack(final_images)
        batch_x['age'] = np.stack([x['age'] for x in data])
        batch_x['country'] = np.stack([x['country'] for x in data])
        # batch_x['intensity'] = np.stack([x['intensity'] for x in data])
        batch_y = np.stack([x['intensity'] for x in data])
        return batch_x, batch_y


if __name__ == '__main__':
    # class ARGS(object):
    #     def __init__(self):
    #         self.dataset_folder_path = './dataset/abaw5/'
    #         self.input_image_size = 299
    # args = ARGS()
    # abaw = ABAWDataset(args, 0)
    # collate_fn = Collator()
    # train_loader = DataLoader(dataset=abaw,
    #                           batch_size=2,
    #                           num_workers=0,
    #                           collate_fn=collate_fn,
    #                           shuffle=True)
    # for batch in train_loader:
    #     print(batch)


    dataset = ABAWDataModule(dataset_folder_path="./dataset/abaw5/",
                             batch_size=4, 
                             input_image_size=299,
                             )
    print('train')
    print(len(dataset.train_dataloader))
    print('val')
    print(len(dataset.val_dataloader))
    print('test')
    print(len(dataset.test_dataloader))


