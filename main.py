import logging
import os
import sys
from pathlib import Path

import nni
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nni.utils import merge_parameter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import albumentations as A

from CustomDataset import SemiSupervisedDataset
from Segnet import Segnet
from Unet import UNet
from dice_loss import MultiClassDiceLoss, MultiClassDiceCoeff

import warnings

warnings.filterwarnings("ignore")

data_dir = Path("./datas")
data_dir.mkdir(parents=True, exist_ok=True)
result_dir = Path("./results")
result_dir.mkdir(parents=True, exist_ok=True)

# 记录最好的模型和验证精度
best_model = None
best_iou = 0.0

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if os.path.exists('log.log'):
    os.remove('log.log')


class EMA():
    def __init__(self, model, decay=0.98):
        self.model = model
        self.decay = decay
        self.shadow = {}  # 记录教师模型的参数
        self.backup = {}  # 备份学生模型的参数

    def register(self):
        # 开始训练前就要调用，将学生模型的参数都记录到shadow里面去
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = param.data.clone()

    def update(self):
        # 把学生的参数更新到teacher模型里面去
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # 把shadow的参数copy到model中
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        # 模型参数恢复为学生模型的参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def setup_logger(logger_name, file_name):
    # 创建 logger 对象
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 创建文件处理器，并设置日志级别和文件名
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)

    # 将文件处理器添加到 logger 对象中
    logger.addHandler(file_handler)

    return logger


# 使用函数设置日志记录器
logger = setup_logger('informations', 'log.log')

# 存储训练过程状态
training_loss_history = []
validation_IoU_history = []
validation_DSC_history = []


# 定义训练函数
def train(args, model, ema, device, train_loader, optimizer, epoch):
    ce_func = torch.nn.CrossEntropyLoss()  # 用于计算模型预测值和标签值之间的距离
    dl_func = MultiClassDiceLoss(num_classes=2, skip_bg=True)  # 有标签的分割dice损失
    mse_func = torch.nn.MSELoss()  # 一致性损失
    epoch_training_loss = []  # 初始化损失列表
    model.train()
    for batch_idx, (images, masks, ys) in enumerate(train_loader):
        images = torch.tensor(images, dtype=torch.float32).to(device)  # （N,3,H,W）
        masks = torch.tensor(masks, dtype=torch.int64).to(device)  # (N,H,W)
        ys = torch.tensor(ys, dtype=torch.int32).to(device)  # (N)
        # 将图像输入student模型中，执行inference
        student_outputs = model(images)
        # 计算损失函数
        if torch.count_nonzero(ys) > 0:  # minibatch是否存在有标签数据
            keep = torch.where(ys > 0)  # 只有有标签的图像才需要算ce loss
            keep_outputs = student_outputs[keep]  # (N_labeled, H, W)
            keep_masks = masks[keep]
            ce_loss = ce_func(keep_outputs, keep_masks)
            dice_loss = dl_func(keep_outputs, keep_masks)
        else:
            ce_loss = 0.0
            dice_loss = 0.0

        # 将图像输入teacher模型中，执行inference,不管是否有标签都要计算mse损失
        ema.apply_shadow()
        teacher_outputs = model(images)
        ema.restore()
        consistent_loss = mse_func(teacher_outputs, student_outputs)

        total_loss = ce_loss + dice_loss + consistent_loss

        optimizer.zero_grad()
        total_loss.backward()

        # EMA更新教师的参数
        ema.update()

        if batch_idx % 100 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss.item()))
        epoch_training_loss.append(total_loss.item())  # 将每个batch的损失添加到当前epoch的损失列表中
    # 计算当前epoch的平均训练损失并添加到训练损失历史列表中
    epoch_avg_training_loss = np.mean(epoch_training_loss)
    training_loss_history.append(epoch_avg_training_loss)


# 定义测试和验证函数
def test(model, device, test_loader):
    print('\n starting evaluating...')
    logger.info('\n starting evaluating...')
    ce_func = torch.nn.CrossEntropyLoss()  # 用于计算模型预测值和标签值之间的距离
    dl_func = MultiClassDiceLoss(num_classes=2, skip_bg=True)  # 有标签的分割dice损失
    dsc_func = MultiClassDiceCoeff(num_classes=2, skip_bg=True)
    mse_func = torch.nn.MSELoss()  # 一致性损失
    global best_model, best_iou
    model.eval()
    eval_dsc = 0.0
    with torch.no_grad():
        for batch_idx, (ims, masks, ys) in enumerate(test_loader):
            ims = ims.to(device).float()
            masks = masks.to(device).long()

            preds = model(ims)
            dsc = dsc_func(preds, masks)
            eval_dsc += dsc.item()

            if batch_idx % 100 == 0:
                print(dsc)

    eval_dsc = eval_dsc / len(test_loader)

    logger.info('\nTest set: Average DSC: {:.20f}%'.format(eval_dsc))
    validation_DSC_history.append(eval_dsc)
    # Save the best model
    if eval_dsc > best_iou:
        best_iou = eval_dsc
        best_model = model.state_dict()  # Save the best model parameters
        torch.save(best_model, result_dir / 'best_model.pth')

    return eval_dsc


# 配置超参数搜索空间
def get_params():
    parser = argparse.ArgumentParser(description='PyTorch Brain Segentation')

    parser.add_argument("--batch_size", type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument("--hidden_size", type=int, default=32, metavar='N', help='hidden layer size (default: 32)')
    parser.add_argument("--lr", type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument("--momentum", type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument("--epochs", type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')

    parser.add_argument("--seed", type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--no_cuda", action='store_true', default=False, help='disables CUDA training')

    args, _ = parser.parse_known_args()
    return args


def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    logger.info(device)

    training_transforms = A.Compose([A.HorizontalFlip(p=0.5),
                                     A.RandomBrightnessContrast(p=0.2),
                                     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=180, p=0.5),
                                     A.GridDistortion(p=0.1),
                                     A.OpticalDistortion(p=0.1),
                                     A.Resize(args['hidden_size'], args['hidden_size'])])

    validation_transform = A.Compose([A.Resize(height=56, width=56, p=1)], p=1)

    # 测试数据路径
    data_roots = ['./datas/thymoma', './datas/thymoma_unlabeled']

    # 创建训练和验证数据集
    training_dataset = SemiSupervisedDataset(data_roots, transform=training_transforms)
    validation_dataset = SemiSupervisedDataset(['./datas/thymoma'], transform=validation_transform)

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args['batch_size'], shuffle=True, **kwargs)

    model = UNet(in_channels=3, out_channels=2)
    model.to(device)
    ema = EMA(model, 0.95)
    ema.register()
    logger.info(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, ema, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader)
        nni.report_intermediate_result(test_acc)

    nni.report_final_result(test_acc)

    # 绘制训练和验证损失

    id = nni.get_trial_id()

    plt.plot(training_loss_history, label='Training Loss')
    plt.plot(validation_DSC_history, label='Validation DSC')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/DSC(%)')
    plt.title('Training and Validation Loss & IoU')
    plt.legend()
    plt.savefig(result_dir / f"result_{id}.png")  # 保存图形到结果文件夹
    plt.close()


if __name__ == '__main__':
    try:
        # 其他代码部分
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)

    except Exception as exception:
        logger.exception(exception)
        raise


