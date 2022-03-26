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
parser.add_argument('--data', default="../data/dataset",
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--t', default=1.0, type=float,help="the same as temperature in Knowledge Distillation")
args = parser.parse_args()


class FairTrain:
    def __init__(self, model, args, save_name=None):

        self.class_feature = None
        self.device = torch.device(args.gpu)
        self.args = args
        self.model = model.to(self.device)

        train_loader, valid_loader, test_loader = get_cifar10_loader(root=args.data, batch_size=args.batchsize)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        if save_name is not None:
            self.save_name = save_name
        else:
            self.save_name = "None.pth"

    def main_train(self):
        args = self.args

        optimizer = torch.optim.SGD(self.model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            # train for one epoch
            self.train(optimizer, epoch, args)

            self.eval_test()

            if (epoch + 1) % 50 == 0:
                a = self.eval_fair()
                print("Confusion Matrix")
                print(a)
                print("=====" * 10)


        save_name = "alpha{}_beta_{}_T_{}".format(args.alpha,args.beta,args.t) + self.save_name
        torch.save(self.model.state_dict(), os.path.join("pth", save_name))

    def train(self, optimizer, epoch, args):
        self.model.train()
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.KLDivLoss()
        criterion3 = nn.KLDivLoss()

        for i, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.long().to(self.device)
            feature, outputs = self.model(data)

            self.update_feature(feature, target)
            loss2_target = self.get_target(target)
            loss3_target = self.get_neg_target(feature,target)

            optimizer.zero_grad()
            loss1 = criterion1(outputs, target)
            loss2 = criterion2(F.log_softmax(feature/args.t, dim=1), F.softmax(loss2_target/args.t, dim=1))
            loss3 = criterion3(F.log_softmax(feature/args.t,dim=1),F.softmax(loss3_target/args.t,dim=1))
            loss = loss1 + args.alpha * loss2 - args.beta * loss3

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(self.train_loader.dataset),
                           100. * i / len(self.train_loader), loss.item()))

    def update_feature(self, feature, target, m=0.999):
        """
        根据当前batch中的label，更新模型的整体class的feature。
        """
        with torch.no_grad():
            target = target.detach().clone()
            feature = feature.detach().clone()
            batch_size = target.size()[0]
            if self.class_feature is None:
                self.class_feature = torch.zeros([10, 512]).to(self.device)
                for i in range(10):
                    class_id = (torch.zeros(batch_size) + i).to(self.device)
                    mask = (class_id == target)
                    sum_i = mask.sum().item()
                    mask = mask.unsqueeze(1)
                    feature_i = torch.sum((feature * mask / sum_i), dim=0) / sum_i
                    self.class_feature[i] = feature_i
            else:
                for i in range(10):
                    class_id = (torch.zeros(batch_size) + i).to(self.device)
                    mask = (class_id == target)
                    sum_i = mask.sum().item()
                    mask = mask.unsqueeze(1)
                    feature_i = torch.sum((feature * mask / sum_i), dim=0) / sum_i
                    self.class_feature[i] = (1 - m) * feature_i + m * self.class_feature[i]

    def get_target(self, target):
        """
        输入真实标签，返回 class_feature对应的 feature 作为KL散度的label
        """
        target = target.detach().clone()
        kl_target = []
        for t in target:
            kl_target.append(self.class_feature[t].unsqueeze(0))
        kl_target = torch.cat(kl_target, dim=0)
        return kl_target

    def get_neg_target(self, feature, target):
        # input:feature:(B,512),class_feature:(10,512)
        # output: (B,10)-->(B)-->(B,512) use self.get_target
        # 在feature上首先计算每个Batch中每个样本与10个feature的相似度
        # 然后再对应的真实类上减去一个比较大的值
        # 然后获取输出最大的标签，也就是除了真实类之外最相似的类。然后获得标签。
        # 获得与target对应的索引的onehot形式tensor，shape为(B,10)
        neg = - 999 * torch.eye(10).to(self.device)
        neg_v = neg.index_select(0,target)

        feature = feature.detach().clone()
        with torch.no_grad():
            # sim : (B,10)
            sim = torch.mm(feature, self.class_feature.t()) + neg_v
            # get max index : (B)
            _, max_sim_index = torch.max(sim, 1)
        max_sim_index = max_sim_index.to(self.device)
        neg_target = self.get_target(max_sim_index)
        return neg_target



    def eval_test(self):
        """
        评估模型的整体准确率
        """
        self.model.eval()
        train_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.long().to(self.device)
                _, outputs = self.model(data)
                train_loss += F.cross_entropy(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        train_loss /= len(self.test_loader.dataset)
        print('Testing: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, total, 100. * correct / total))
        training_accuracy = correct / total
        return train_loss, training_accuracy

    def eval_fair(self):
        """
        评估模型的公平性
        :return: 混淆矩阵
        """
        self.model.eval()
        confusion_m = np.zeros((10, 10))
        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                _, outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)

                for i, j in zip(target, predicted):
                    confusion_m[i][j] += 1
        return confusion_m

    def eval_fgsm_robustness(self,epsilon):
        self.model.eval()
        correct = 0
        total = 0
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            _, outputs = self.model(data)
            total += target.size(0)
            self.model.zero_grad()
            loss = F.nll_loss(outputs, target)
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = data + epsilon * data_grad.sign()
            perturbed_data = torch.clamp(perturbed_data, 0, 1)

            _, final_outputs = self.model(perturbed_data)
            _, final_predicted = torch.max(final_outputs.data, 1)
            correct += (final_predicted == target).sum().item()

        final_acc = correct / total
        print("FGSM  Epsilon: {} \t Test Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))
        return final_acc

    def eval_pgd_robustness(self, epsilon,alpha,iters):
        self.model.eval()
        correct = 0
        total = 0
        loss_CE = nn.CrossEntropyLoss()

        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            total += target.size(0)
            ori_images = data.data
            for i in range(iters):
                data.requires_grad = True
                _, outputs = self.model(data)
                self.model.zero_grad()
                loss = loss_CE(outputs,target)
                loss.backward()
                adv_data = data + alpha * data.grad.sign()
                eta = torch.clamp(adv_data - ori_images, min=-epsilon, max=epsilon)
                data = torch.clamp(ori_images+eta, 0, 1).detach_()

            _, final_outputs = self.model(data)
            _, final_predicted = torch.max(final_outputs.data, 1)
            correct += (final_predicted == target).sum().item()

        final_acc = correct / total
        print("PGD  Epsilon: {} Alpha: {} iters: {} \t Test Accuracy = {} / {} = {}".format(epsilon,alpha,iters, correct, total, final_acc))
        return final_acc


if __name__ == "__main__":
    model = ResNet18()
    fair = FairTrain(model=model, args=args, save_name="test.pth")
    fair.main_train()
