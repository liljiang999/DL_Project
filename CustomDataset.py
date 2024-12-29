import glob

import cv2
import torch
import os
from torchvision import transforms
from PIL import Image
import random
import pandas as pd
import os.path as osp
import numpy as np


# 创建半监督数据集
class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, data_roots, transform=None, image_ext=".jpg"):
        self.data_roots = data_roots
        self.transform = transform
        self.image_ext = image_ext

        self.samples = []

        for root in data_roots:
            im_files = glob.glob(osp.join(root, "*" + self.image_ext), recursive=True)
            for im_file in im_files:
                anno_file = im_file.replace(self.image_ext, "_mask"+ self.image_ext)
                if osp.exists(anno_file):
                    self.samples.append((im_file, anno_file))
                else:
                    self.samples.append((im_file, None))

        np.random.shuffle(self.samples)

        print('Total samples:', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        im_file, anno_file = self.samples[idx]
        # 用cv2读取BGR
        image = cv2.imread(im_file, 1)
        # 转为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if anno_file is None:
            mask = np.zeros(image.shape[0:2], dtype=np.int32)  # 创建0矩阵, 后面也不会用到
            y = 0.0  # 指示minibatch里面哪张图像无标签
        else:
            mask = cv2.imread(anno_file, 0)  # 灰度图方式读取
            mask[mask > 0] = 1
            y = 1.0  # 指示minibatch里面哪张图像有标签

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = image.transpose((2, 0, 1))  # (H,W,3)->(3,H,W)
        return image, mask, y

# 测试代码
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import albumentations as A

    transforms = A.Compose([A.Resize(height=128, width=128, p=1)], p=1)

    # 测试数据路径
    data_roots = ['./datas/thymoma_unlabeled', './datas/thymoma']
    # 创建数据集对象
    dataset = SemiSupervisedDataset(data_roots, transform=transforms)

    for k in range(len(dataset)):
        image, mask, y = dataset[k]
        print(image.shape, mask.shape)
        print(np.unique(mask))

        image = image.transpose((1, 2, 0))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')

        plt.show()
