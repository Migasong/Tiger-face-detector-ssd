import pickle as pkl
import numpy as np
import settings
import os

def readtxt(train_or_val):
    model_dir = settings.config_dir + '/dataset'
    if train_or_val == 'train':
        txt_file = os.path.join(model_dir, 'lb.txt')
        save_name = 'tiger.pkl'
    elif train_or_val == 'val':
        txt_file = os.path.join(model_dir, 'wider_face_val_bbx_gt.txt')
        save_name = 'val.pkl'

    print(txt_file)
    with open(txt_file) as f:
        lines = f.readlines()
    print(len(lines))

    database = {'image_name':[], 'label':[],  'num':[]}
    i = 0
    n = 1

    while i < len(lines):
        #if i > 6:
        #    break
        image_name = lines[i].split('\n')[0]
        num = int(lines[i+1])
        if num == 0:
            i = i + 2
            continue

        database['image_name'].append(image_name)
        database['num'].append(num)
    
        label = np.empty(shape=(num, 5))
    
        for k in range(num):
            label_tmp = (lines[i+2+k].split(' '))
            label[k, 4] = 0 
            for kk in range(4):
                label[k, kk] = int(label_tmp[kk])
    
        database['label'].append(label)
        print('No.', n, 'image_name:', lines[i], '\tnum:', num, 'label.shape:', label.shape)
        i = i + 2 + num
        n = n + 1

    print('\nimage_name:', len(database['image_name']), '\tnum:', len(database['num']), '\tlabel:', len(database['label']))
    
    with open(os.path.join(model_dir, save_name), 'wb') as fw:
        pkl.dump(database, fw)
    
    return database

if __name__=='__main__':
    database = readtxt('train')
    for i in range(1):
        print('\n', database['image_name'][i], database['num'][i], database['label'][i].shape)
    





