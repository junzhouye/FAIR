"""
Fairness_train_V1
最初之作

loss1: 就是正常分类使用的交叉熵损失函数
loss2: 根据对比学习使用的损失函数修改得到。就是使用MoCo这篇论文修改得来的。
这里需要注意的是，MoCo是对比学习，也就是无监督学习。它使用相同样本的经过不同编码器的feature作为正样本，其余全部作为负样本，经过矩阵乘法获得
(batch_size,all_sample_number)的矩阵，其中batch_size是正样本的数量，all_sample_number是负样本的数量

我这里使用的是一个batch的样本，batch设置为128
self_to_self 作为正样本，也是生成的labels。而负样本则是其他。

感觉到取材自MoCo的这个损失函数存在其局限性。它使用的损失函数是使用在分类任务中的。
"""

import torch
from data.cifar10 import get_cifar10_loader
import argparse
import torch.optim as optim
import random
import torch.nn as nn
import torch.nn.functional as F
from models.cifar10_resnet import ResNet18
import os
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default="./data/dataset",
                    help='path to dataset')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batchsize', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1, type=float)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.gpu is not None:
        device = torch.device(args.gpu)

    main_worker(device, args)


def main_worker(device, args):
    model = ResNet18().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    train_loader, valid_loader, test_loader = get_cifar10_loader(root=args.data, batch_size=args.batchsize)

    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, valid_loader, model, device, optimizer, epoch, args)

        eval_test(model, device, test_loader)

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(),
                       os.path.join("pth", 'FairTrainV1_alpha{}_epoch{}.pth'.format(args.alpha, epoch + 1)))


def train(train_loader, valid_loader, model, device, optimizer, epoch, args):
    # eval_res guide mask
    eval_res = eval_fair(model, valid_loader)

    model.train()
    criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion2 = nn.CrossEntropyLoss(label_smoothing=0.1)
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.long().to(device)
        feature, outputs = model(data)
        optimizer.zero_grad()
        loss1 = criterion1(outputs, target)
        feature_logit = torch.mm(feature, feature.t())
        labels = torch.arange(data.size()[0], dtype=torch.long).to(device)
        # nn.CrossEntropyLoss have softmax
        loss2 = criterion2(feature_logit, labels)
        loss = loss1 + args.alpha * loss2
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))


def eval_test(model, device, test_loader):
    model.eval()
    train_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.long().to(device)
            _, outputs = model(data)
            train_loss += F.cross_entropy(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    train_loss /= len(test_loader.dataset)
    print('Testing: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, total, 100. * correct / total))
    training_accuracy = correct / total
    return train_loss, training_accuracy


if __name__ == "__main__":
    main()
