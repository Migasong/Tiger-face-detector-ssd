import os
import torch
import cv2
import numpy as np
from torch import nn
from torch.nn import DataParallel
from torch.autograd import Variable
from dataset import testDataset
from ssd import build_ssd
from config import voc
from box_utils import decode, nms
from train import Session
import settings

class tiger_test():
    def __init__(self):
        self.dataset = testDataset()
        self.cfg = voc 
        self.variance = voc['variance']
        self.conf_thresh = 0.2
        self.nms_thresh = 0.5
        self.top_k = 200
        self.model_dir = settings.model_dir
        self.net = build_ssd('train', self.cfg['min_dim'], self.cfg['num_classes'])
        
        if settings.num_GPU > 1:
            self.net = DataParallel(self.net,
            device_ids=list(range(settings.num_GPU)))
        else:
            self.net = self.net.cuda()

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            return
        self.net.load_state_dict(obj['network'])

    def test(self):
        size = self.dataset.__len__()
        self.load_checkpoints('latest')
        
        result = {'img_name':[], 'loc':[], 'scores':[]}
        for i in range(size):
            sample = self.dataset.__getitem__(i)
            img = sample['data']
            img = img.unsqueeze(0)
            scale = sample['scale']
            img = Variable(img.cuda())
            
            pred = self.net(img)
            loc_tmp = pred[0].data.squeeze(0)
            conf_tmp = 1 / (1 + torch.exp(-pred[1].data.squeeze(0)))
            priors_tmp = (pred[2].data).type(type(img.data))
            
            decoded_boxes = decode(loc_tmp, priors_tmp, self.variance)
            c_mask = conf_tmp.gt(self.conf_thresh)
            scores = conf_tmp[c_mask]
            
            if scores.dim() == 0:
                continue
            
            l_mask = c_mask.expand_as(decoded_boxes)
            bboxes = decoded_boxes[l_mask].view(-1, 4)
            
            # idx of highest scoring and non-overlapping boxes per class
            ids, count = nms(bboxes, scores, self.nms_thresh, self.top_k)
            scores = scores[ids[:count]]
            loc = bboxes[ids[:count]] / scale

            result['img_name'].append(sample['name'])
            result['loc'].append(loc)
            result['scores'].append(scores)


        return result


if __name__== '__main__':
    
    test = tiger_test()
    result = test.test()
    for i in range(len(result['img_name'])):
        print('\n\nNo.', i)
        print('img name:\t', result['img_name'][i])
        print('location:\t', result['loc'][i].cpu().numpy())
        print('conf:\t\t', result['scores'][i].cpu().numpy())

        img_dir = os.path.join(settings.test_dir, result['img_name'][i])
        img = cv2.imread(img_dir)
        loc = result['loc'][i].cpu().numpy()
        for l in range(loc.shape[0]):
            x1 = loc[l][0]
            y1 = loc[l][1]
            x2 = loc[l][2]
            y2 = loc[l][3]
            draw = cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
            save_path = settings.config_dir + '/output/' + result['img_name'][i]
            cv2.imwrite(save_path, draw)
    
    print('the number of picture:', len(result['img_name']))








    

