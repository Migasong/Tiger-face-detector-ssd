import os
import csv
import cv2
import itertools
import torch
import numpy as np
import pandas as pd
import readtxt
import settings
import readtxt
import pickle as pkl
from torch.utils.data import Dataset, DataLoader


class WiderfaceDataset(Dataset):
    def __init__(self, train_or_val):
        super(WiderfaceDataset, self).__init__()
        if not os.path.exists(os.path.join(settings.config_dir, 'dataset/tiger.pkl')):
            database = readtxt('train')
        else:
            with open(os.path.join(settings.config_dir, 'dataset/tiger.pkl'), 'rb') as fr:
                database = pkl.load(fr)
        
        
        self.size = len(database['image_name'])
        
        train_img = []
        train_label =[]
        train_num = []
        val_img = []
        val_label =[]
        val_num = []

        for i in range(self.size):
            if (i % 5 == 0):
                val_img.append(database['image_name'][i])
                val_label.append(database['label'][i])
                val_num.append(database['num'][i])
            else:
                train_img.append(database['image_name'][i])
                train_label.append(database['label'][i])
                train_num.append(database['num'][i])

        if train_or_val == 'train':
            self.num = int(len(train_img))
            self.img_lst = train_img
            self.label_lst = train_label
            self.num_lst = train_num

        elif train_or_val == 'val':
            self.num = int(len(val_img))
            self.img_lst = val_img
            self.label_lst = val_label
            self.num_lst = val_num

    def __len__(self):
        return self.num
        
    def __getitem__(self, idx):
        
        img_dir = os.path.join(settings.data_dir, self.img_lst[idx])
        #print(self.img_lst[idx])
        if not os.path.exists(img_dir):
            print('Did not exist such an image!')
            print(img_dir)
        
        img_original = cv2.imread(img_dir)
        
        #img = img_original
        img = img_padding(img_original)[0]
        scale = img_padding(img_original)[1]

        img = img.astype('float32')
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        label = self.label_lst[idx]
        """
        label_tmp = self.label_lst[idx] 
        label = np.empty(label_tmp.shape)
        label[:, 0:2] = label_tmp[:, 0:2]
        label[:, 4] = label_tmp[:, 4]
        label[:, 2] = label_tmp[:, 0] + label_tmp[:, 2] 
        label[:, 3] = label_tmp[:, 1] + label_tmp[:, 3] 
        """
        #print('label', label_tmp[:, 0] + label_tmp[:, 2])
    
        label[:, 0:4] = label[:, 0:4] * scale
        label = label.astype('float32')

        #print('label_tmp', label_tmp[:, 0:4])

        lb_num = self.num_lst[idx]
        #shape = torch.from_numpy(np.array([img_original.shape[0], img_original.shape[1]]))
        
        sample = {'name': self.img_lst[idx], 'data': img, 'label': label, 'lb_num': lb_num}

        return sample


class testDataset(Dataset):
    def __init__(self): 
        super(testDataset, self).__init__()
        self.test_dir = settings.test_dir
        self.file_name = os.listdir(self.test_dir)

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        
        img_dir = os.path.join(self.test_dir, self.file_name[idx])

        #print(self.img_lst[idx])
        if not os.path.exists(img_dir):
            print('Did not exist such an image!')
            print(img_dir)
        
        img_original = cv2.imread(img_dir)
        shape = img_original.shape
        
        #img = img_original
        img = img_padding(img_original)[0]
        scale = img_padding(img_original)[1]

        img = img.astype('float32')
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        sample = {'name':self.file_name[idx], 'data': img, 'scale': scale, 'shape': shape}

        return sample

def img_padding(imgs):
    h, w = imgs.shape[0],imgs.shape[1]
    if h > w:
        sz = h
    else:
        sz = w
    bottom, right = sz - h, sz - w
        
    imgs = cv2.copyMakeBorder(imgs, 0, bottom, 0, right, cv2.BORDER_CONSTANT)
        
    scale = 300 / imgs.shape[0]        
    imgs = cv2.resize(imgs, (300, 300))
    out = [imgs, scale]
    return out

def Tensor2Img(img_Tensor):
    imgs = img_Tensor.numpy()
    #print('1',imgs.shape)
    imgs = imgs.transpose(1, 2, 0)
    #print('2',imgs.shape)
    imgs = imgs.astype('uint8')
    return imgs

if __name__=='__main__':
    """
    # trainDataset
    dataset = WiderfaceDataset('train')      
    print(dataset.__len__())
    for i in range(2):
        sample = dataset.__getitem__(i)
        print(sample['label'])
        
        img = Tensor2Img(sample['data'])
        name = sample['name']
        label = sample['label']
        x1 = label[0,0]
        x2 = label[0,2]
        y1 = label[0,1]
        y2 = label[0,3]
        print('img shape', sample['data'].shape)
        draw = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imshow(name, draw)
        cv2.waitKey(0)
        
    for i in range(1):
        print('NO.', i)
        
        data = dataset[i]
        img = data['data']
        
        path = os.path.join(settings.data_dir, data['name'])
        print('path:\t\t', path)
        img_before = cv2.imread(os.path.join(settings.data_dir, data['name']))
        
        print('img shape:\t', img.shape)
        print('label num:\t', data['lb_num'])
        print('label:\t\t', data['label'], '\n')
    """    
    ## testDataset
    dataset = testDataset()
    print('dataset size:\t', dataset.__len__())
    for i in range(2):
        sample = dataset.__getitem__(i)
        print('\nNO.', i)
        print('img name:\t', sample['name'])
        print('original shape:', sample['shape'])
        print('resize shape:\t', sample['data'].shape)
        print('scale:\t\t', sample['scale'])
    







