import logging
import os
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


from CustomDataset import SemiSupervisedDataset
from Segnet import Segnet

data_dir = Path("./datas")
data_dir.mkdir(parents=True, exist_ok=True)
result_dir = Path("./results")
result_dir.mkdir(parents=True, exist_ok=True)


if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if os.path.exists('log.log'):
    os.remove('log.log')


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
validation_loss_history = []
validation_accuracy_history = []

# 定义训练函数
def train(args, model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_training_loss = []  # 初始化损失列表
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        epoch_training_loss.append(loss.item())  # 将每个batch的损失添加到当前epoch的损失列表中
    # 计算当前epoch的平均训练损失并添加到训练损失历史列表中
    epoch_avg_training_loss = np.mean(epoch_training_loss)
    training_loss_history.append(epoch_avg_training_loss)


# 定义测试和验证函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    validation_loss_history.append(test_loss)
    validation_accuracy = 100. * correct / len(test_loader.dataset)
    validation_accuracy_history.append(validation_accuracy)

    return validation_accuracy


# 配置超参数搜索空间
def get_params():
    parser = argparse.ArgumentParser(description='PyTorch Brain Segentation')

    parser.add_argument("--batch_size", type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument("--hidden_size", type=int, default=64, metavar='N', help='hidden layer size (default: 64)')
    parser.add_argument("--lr", type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument("--momentum", type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument("--epochs", type=int, default=5, metavar='N', help='number of epochs to train (default: 10)')

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

    training_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize(args['hidden_size']),
        transforms.CenterCrop(args['hidden_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    validation_transform = transforms.Compose([
        transforms.Resize(args['hidden_size']),
        transforms.CenterCrop(args['hidden_size']),
        transforms.ToTensor(),
    ])

    # 创建训练和验证数据集
    training_dataset = SemiSupervisedDataset(data_dir / "thymoma_labeled", data_dir / "thymoma_unlabeled",
                                     transform=training_transform, image_ext=".jpg")
    validation_dataset = SemiSupervisedDataset(data_dir / "thymoma_labeled", data_dir / "thymoma_unlabeled",
                                       transform=validation_transform, image_ext=".jpg")

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args['batch_size'], shuffle=True, **kwargs)

    model = Segnet(input_nc=3, output_nc=1)
    logger.info(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader)
        nni.report_intermediate_result(test_acc)

    nni.report_final_result(test_acc)

    # 绘制训练和验证损失
    plt.plot(training_loss_history, label='Training Loss')
    plt.plot(validation_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss(%)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(result_dir / f"training_validation_loss_{id}.png")  # 保存图形到结果文件夹
    plt.close()
    image_urls_input = result_dir / f"training_validation_loss_{id}.png"

    # 绘制验证精度图
    plt.plot(validation_accuracy_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig(result_dir / f"validation_accuracy_{id}.png")  # 保存图形到结果文件夹
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


