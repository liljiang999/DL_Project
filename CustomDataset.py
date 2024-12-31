from torch.utils.data import Dataset
import glob
import os.path as osp
import cv2
import numpy as np
import random


class SemiSupervisedDataset(Dataset):
    def __init__(self, data_roots, transform=None):
        self.transforms = transform
        self.im_mask_files = []

        for root in data_roots:
            print(root)
            mask_files = glob.glob(osp.join(root, '*_mask.jpg'))
            if len(mask_files) == 0:
                print("yes")
                im_files = glob.glob(osp.join(root, '*.jpg'))
                random.shuffle(im_files)  # 打乱
                for im_file in im_files:
                    self.im_mask_files.append([im_file, None])

                    if len(self.im_mask_files) >= 10000:
                        break
            else:
                for mask_f in mask_files:
                    im_f = mask_f.replace('_mask', '')
                    if osp.exists(im_f):
                        self.im_mask_files.append([im_f, mask_f])

                print(len(self.im_mask_files))

        np.random.shuffle(self.im_mask_files)

        print('Found images and masks:', len(self.im_mask_files))

    def __len__(self):
        return len(self.im_mask_files)

    def __getitem__(self, index):
        im_f, mask_f = self.im_mask_files[index]
        im = cv2.imread(im_f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # cv2默认用BGR格式，要转为RGB

        if mask_f is None:
            mask = np.zeros(im.shape[0:2], dtype=np.int32)  # 创建0矩阵, 后面也不会用到
            y = 0.0  # 指示minibatch里面哪张图像无标签
        else:
            mask = cv2.imread(im_f, 0)  # 灰度图方式读取
            mask[mask > 0] = 1
            y = 1.0  # 指示minibatch里面哪张图像有标签

        # 如果你的transforms库报错，那就不要给transforms，用cv2.resize
        '''im = cv2.resize(im, (128, 128))
        mask = cv2.resize(mask, (128, 128))'''
        if self.transforms:
            trans = self.transforms(image=im, mask=mask)
            im = trans['image']
            mask = trans['mask']
        # 切记：要把mask里面的值转为0和1
        mask[mask > 0] = 1
        im = im.transpose((2, 0, 1))  # (C, H, W)
        return im, mask, y


if __name__ == '__main__':
    import albumentations as A
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    transforms = A.Compose([A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.2),
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=180, p=0.5),
                            A.GridDistortion(p=0.1),
                            A.OpticalDistortion(p=0.1),
                            A.Resize(128, 128)])

    dataset = SkinDataset(data_root='../../datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_Data',
                          transforms=transforms)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    for im, mask in dataloader:
        print(im.shape, mask.shape)
        im = im[0, ...].numpy().transpose((1, 2, 0))
        mask = mask[0, ...].numpy()
        plt.subplot(1, 2, 1)
        plt.imshow(im)
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.show()

