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
from trainers.FairTrain import FairTrain

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default="./data/dataset",
                    help='path to dataset')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batchsize', default=256, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=4399, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--t', default=1.0, type=float,help="the same as temperature in Knowledge Distillation")
args = parser.parse_args()


model = ResNet18()
fair = FairTrain(model=model, args=args, save_name="FairTrain.pth")
fair.main_train()