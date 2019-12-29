import os
import os.path as osp
import json

import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import pdb as ipdb


# gt_dir = 'gTBox.csv'
# train_dir = 'positive image set'
label_map = ('airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 
             'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle')

vhr = {
    'num_classes': 11,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [5, 10, 20, 40, 60, 80],
    'max_sizes': [ 10, 20, 40, 60, 80, 120],   
    # 'min_sizes': [21, 45, 99, 153, 207, 261],
    # 'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VHR',
}


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


class vhrData(data.Dataset):
    """VHR-10 Dataset for Detection training.
    Args:
        root: Root directory where images are.

    """
    def __init__(self, root, train_dir='../Datasets/VHR-10_dataset/positive image set', 
                 gt_dir='gTBox.json', transform=None, target_transform=None, obj_num=None):
        imgs = []
        bbox = []
        
        # data_pd = pd.read_csv(osp.join(root, gt_dir))
        # if not obj_num:
        #     for _, row in data_pd.iterrows():
        #         imgs.append(row[0])
        #         bbox.append(row[2])
        # else:
        #     for _, row in data_pd.iterrows():
        #         if row[1] >= obj_num[0] and row[1] <= obj_num[1]:
        #             imgs.append(row[0])
        #             bbox.append(row[2])
        #         else:
        #             continue

        with open (osp.join(root, gt_dir)) as f:
            data_label = json.load(f)
        for key in data_label.keys():
            imgs.append(key)
            bbox.append(data_label[key])
        
        self.imgs = imgs
        self.bbox = bbox
        self.root = root
        self.train_dir = train_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        file_name, bbox = self.imgs[index], self.bbox[index]
        # img = cv2.imread(osp.join(self.train_dir, file_name))
        img = Image.open(osp.join(self.train_dir, file_name)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        # bbox = [i for i in bbox]
        # for target in bbox:
        #     target = tuple([target[4], target[0], target[1], target[2], target[3]])

        return img, bbox
    
    def __len__(self):
        return len(self.imgs)

    def pull_image(self, index, show_box=False):
        """Return the origin image. With GTBox if show_box=True
        Args:
            index: index of photos
            show_box: if True, show GTBox in image
        Return:
            cv2 img
        """
        file_name, bbox = self.imgs[index], self.bbox[index]
        img = cv2.imread(osp.join(self.root, self.train_dir, file_name))

        if show_box:
            for _, pos in enumerate(bbox):
                cv2.rectangle(img, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 4)
        return img


if __name__ == '__main__':
    test_data = vhrData('../../Datasets/VHR-10_dataset')
    print(len(test_data))
    
    img_test = test_data.pull_image(np.random.randint(600), show_box=True)
    plt.imshow(img_test)
    plt.waitforbuttonpress()
