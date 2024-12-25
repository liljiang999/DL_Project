import torch
import os
from torchvision import transforms
from PIL import Image
import random
import pandas as pd
import numpy as np


# 创建半监督数据集
class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, labeled_dir, unlabeled_dir, transform=None, image_ext=".jpg"):
        self.labeled_dir = labeled_dir
        self.unlabeled_dir = unlabeled_dir
        self.transform = transform
        self.image_ext = image_ext

        # 加载有标签图像和掩膜
        self.labeled_images = [f for f in os.listdir(labeled_dir) if f.endswith(self.image_ext)]

        # 加载无标签图像
        self.unlabeled_images = [f for f in os.listdir(unlabeled_dir) if f.endswith(self.image_ext)]

        # 随机打乱无标签图像数据
        random.shuffle(self.unlabeled_images)

    def __len__(self):
        # 半监督数据集的长度是有标签和无标签数据的总和
        return len(self.labeled_images) + len(self.unlabeled_images)

    def __getitem__(self, idx):
        if idx < len(self.labeled_images):
            # 从有标签图像中加载
            img_name = os.path.join(self.labeled_dir, self.labeled_images[idx])
            label_name = img_name.replace(self.image_ext, self.image_ext)

            # 读取图像和标签
            image = Image.open(img_name).convert("RGB")  # 读取彩色图像
            label = Image.open(label_name).convert("L")  # 读取标签图像（灰度图）

            # 如果有指定 transform，则应用
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(np.array(label)).long()  # 转换标签为 tensor

            return image, label  # 有标签返回 (图像, 标签)
        else:
            # 从无标签图像中加载
            idx_unlabeled = idx - len(self.labeled_images)
            img_name = os.path.join(self.unlabeled_dir, self.unlabeled_images[idx_unlabeled])

            # 读取无标签图像
            image = Image.open(img_name).convert("RGB")  # 读取彩色图像

            # 如果有指定 transform，则应用
            if self.transform:
                image = self.transform(image)

            return image, None  # 无标签返回 (图像, None)
