import os
import pandas as pd
from tqdm import tqdm
from time import time
from argparse import ArgumentParser

import torch
from torch import nn
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose

from efficientnet_pytorch import EfficientNet
from adabelief_pytorch import AdaBelief
from torchsummary import summary

from utils import Cutout


class ExperimentCIFAR10:
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        max_epoch=100,
        batch_size=1024,
        cutout=False,
        data_dir='./data',
        csv_dir='results/csv',
        csv_name='record.csv',
        pth_dir='results/pth',
        pth_name='params.pth'
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.cutout = cutout
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.csv_name = csv_name
        self.pth_dir = pth_dir
        self.pth_name = pth_name

        self.device = 'cuda:0' if is_available() else 'cpu'
    
    def __call__(self):
        self.run()

    def prepare_data(self, data_dir, batch_size=1024, cutout=False):
        train_transforms = [ToTensor()]
        if cutout:
            train_transforms.append(Cutout(1, 16))
        train_data = CIFAR10(data_dir, train=True, download=True, transform=Compose(train_transforms))
        test_transforms = [ToTensor()]
        test_data = CIFAR10(data_dir, train=False, download=False, transform=Compose(test_transforms))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    
    def run(self):
        model = self.model.to(self.device)
        optimizer = self.optimizer
        scheduler = self.scheduler

        # パラメータが存在するならロードする.
        if os.path.isfile(os.path.join(self.pth_dir, self.pth_name)):
            print(f'{os.path.join(self.pth_dir, self.pth_name)} already exists.')
            model.load_state_dict(torch.load(os.path.join(self.pth_dir, self.pth_name)))

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.isdir(self.csv_dir):
            os.makedirs(self.csv_dir)
        if not os.path.isdir(self.pth_dir):
            os.makedirs(self.pth_dir)

        loaders = self.prepare_data(self.data_dir, batch_size=self.batch_size, cutout=self.cutout)
        train_loader = loaders[0]
        test_loader = loaders[1]

        for _ in tqdm(range(self.max_epoch)):
            train_loss, train_acc, train_time = self.train(train_loader)
            test_loss, test_acc = self.eval(test_loader)

            # モデルの保存, 上書き
            torch.save(model.to('cpu').state_dict(), os.path.join(self.pth_dir, self.pth_name))
            model.to(self.device)

            # csvへの出力, 上書き
            record = [[train_loss, train_acc, train_time, test_loss, test_acc]]
            df = pd.DataFrame(record, columns=None, index=None)
            if os.path.isfile(os.path.join(self.csv_dir, self.csv_name)):
                df.to_csv(os.path.join(self.csv_dir, self.csv_name), mode='a', header=None, index=None)
            else:
                header = ['train_loss', 'train_acc', 'train_time', 'test_loss', 'test_acc']
                df.to_csv(os.path.join(self.csv_dir, self.csv_name), mode='a', header=header, index=None)

    def train(self, train_loader):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        i = 0
        total = 0
        correct = 0
        running_loss = 0
        tic = time()

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i += 1

        train_loss = running_loss / i
        train_acc = correct / total
        train_time = time() - tic
        return train_loss, train_acc, train_time

    def eval(self, test_loader):
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        i = 0    # ステップ数
        total = 0    # 全てのテストデータの数
        correct = 0    # 正しく分類されたテストデータの数
        running_loss = 0.0  # 予測損失の合計

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                    
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                i += 1

        test_loss = running_loss / i
        test_acc = correct / total
        return test_loss, test_acc


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--csv_dir', type=str, default='results/csv')
    parser.add_argument('--pth_dir', type=str, default='results/pth')
    parser.add_argument('--optimizer', type=str, default='momentum')
    args = parser.parse_args()

    opt_dict = {
        'sgd': (SGD, {'lr': 1e-3}),
        'momentum': (SGD, {'lr': 1e-3, 'momentum': 0.9}),
        'adam': (Adam, {'lr': 1e-3}),
        'amsgrad': (Adam, {'lr': 1e-3, 'amsgrad': True}),
        'adabelief': (AdaBelief, {'lr': 1e-3}),
    }

    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 10)
    summary(model, [3, 32, 32])

    optimizer = opt_dict[args.optimizer][0](model.parameters(), **opt_dict[args.optimizer][1])
    print(optimizer)

    csv_name = f'{args.optimizer}.csv'
    if args.cutout:
        csv_name = csv_name[0:-4] + '+cutout.csv'
 
    pth_name = f'{args.optimizer}.pth'
    if args.cutout:
        pth_name = pth_name[0:-3] + '+cutout.pth'

    experiment = ExperimentCIFAR10(
        model = model,
        optimizer = optimizer,
        scheduler=None,
        max_epoch = args.max_epoch,
        batch_size = args.batch_size,
        cutout = args.cutout,
        csv_dir = args.csv_dir,
        csv_name = csv_name,
        pth_dir = args.pth_dir,
        pth_name = pth_name,
    )

    experiment()
