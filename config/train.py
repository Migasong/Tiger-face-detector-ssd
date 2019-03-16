import os
import sys
import time
import logging
import argparse
import numpy as np
import pickle
import cv2
import itertools
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, DataParallel
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from scipy.spatial.distance import pdist
from scipy.special import expit

import settings
from dataset import WiderfaceDataset
from ssd import build_ssd
from config import voc
from multibox_loss import MultiBoxLoss

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(1)
#torch.cuda.set_device(settings.device_id)



def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        self.step = 0
        self.cfg = voc
        self.net = build_ssd('train', self.cfg['min_dim'], self.cfg['num_classes'])
        if settings.num_GPU > 1:
            self.net = DataParallel(self.net, 
                    device_ids=list(range(settings.num_GPU)))
        else:
            self.net = self.net.cuda()    
            #self.net = self.net
        
        self.save_steps = settings.save_steps

        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}


    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def start(self):
        self.save_checkpoints('latest')

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)

        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(' '.join(outputs))

        if self.step % self.save_steps == self.save_steps - 1:
            self.save_checkpoints('latest')
            self.save_checkpoints('step_%d' % self.step)

    def get_dataloader(self, train_or_val):
        dataset = WiderfaceDataset(train_or_val)
        self.dataloaders = \
                       DataLoader(dataset, batch_size=self.batch_size, 
                             shuffle=True, num_workers=self.num_workers)
        return iter(self.dataloaders)

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'network': self.net.state_dict(),
            'clock': self.step
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            return
        self.net.load_state_dict(obj['network'])
        self.step = obj['clock']


def run_train_val(ckp_name='latest'):
    sess = Session() 
    train_tb = sess.tensorboard('train')
    val_tb = sess.tensorboard('val')
    
    #crit = MSELoss()
    crit = MultiBoxLoss(voc['num_classes'], 0.5, True, 0, True, 3, 0.5, False, True)

    opt = Adam(sess.net.parameters(), lr=settings.lr)

    dts = {
        'train': sess.get_dataloader('train'),
        'val': sess.get_dataloader('val'),
    }

    sess.load_checkpoints(ckp_name)

    def inf_batch(name):
        try:
            batch = next(dts[name])
        except StopIteration:
            dts[name] = sess.get_dataloader(name)
            batch = next(dts[name])
        
        data, label = batch['data'], batch['label']
        data = data.cuda()
        data, label = Variable(data), Variable(label)
        
        
       # data = torch.squeeze(data)
       # label = torch.squeeze(label, 0) 
       # label = label.numpy()
        pred = sess.net(data)
       # print('ground truth label', label)
        loss_l, loss_c = crit(pred, label)

        #print('loss_l', loss_l, 'loss_c', loss_c)
        loss = loss_l + loss_c
        err = loss.data[0]
        sess.write(name, {'loss': err, 'err': err})
        
        return loss

    while True:
        sess.net.zero_grad()
        loss = inf_batch('train')
        loss.backward()
        opt.step()

        loss = inf_batch('val')

        sess.step += 1

"""
def run_test(ckp_name):


    dt = FashionDataset('val')
    dt = DataLoader(dt, batch_size=settings.batch_size, 
            shuffle=False, num_workers=settings.num_workers)

    all_num = 0
    err_num = 0
    sess = Session()
    sess.load_checkpoints(ckp_name)
    for i, batch in enumerate(dt):
        data, label = batch['data'], batch['label']
        data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        pred = sess.net(data)
        batch_err = error_rate(pred['prob'], label, 0)
        logger.info('batch %d error rate: %f' % (i, batch_err))
        err_num += batch_err
        all_num += 1

    logger.info('total error rate: %f' % (err_num / all_num))
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', default='train')
    parser.add_argument('-m', '--model', default='latest')
    args = parser.parse_args(sys.argv[1:])
    
    if args.action == 'train':
        run_train_val(args.model)
        #run_train_val(args.model)
    elif args.action == 'test':
        run_test(args.model)

    






