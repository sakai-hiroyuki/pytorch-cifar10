import os
import pandas as pd
from tqdm import tqdm
from time import time

import torch
from torch import nn
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose

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
        pth_name='params.pth',
        use_tqdm=True
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
        self.use_tqdm = use_tqdm

        self.device = 'cuda:0' if is_available() else 'cpu'
    
    def __call__(self):
        self.run()

    def prepare_data(self):
        train_transforms = [ToTensor()]
        if self.cutout:
            train_transforms.append(Cutout(1, 16))
        train_data = CIFAR10(self.data_dir, train=True, download=True, transform=Compose(train_transforms))
        
        test_transforms = [ToTensor()]
        test_data = CIFAR10(self.data_dir, train=False, download=False, transform=Compose(test_transforms))
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
    
    def run(self):
        self.model = self.model.to(self.device)

        # パラメータが存在するならロードする.
        if os.path.isfile(os.path.join(self.pth_dir, self.pth_name)):
            print(f'{os.path.join(self.pth_dir, self.pth_name)} already exists.')
            self.model.load_state_dict(torch.load(os.path.join(self.pth_dir, self.pth_name)))

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.isdir(self.csv_dir):
            os.makedirs(self.csv_dir)
        if not os.path.isdir(self.pth_dir):
            os.makedirs(self.pth_dir)

        loaders = self.prepare_data()
        train_loader = loaders[0]
        test_loader = loaders[1]

        epochs = range(self.max_epoch)
        if self.use_tqdm:
            epochs = tqdm(epochs)
        for _ in epochs:
            train_loss, train_acc, train_time = self.train(train_loader)
            test_loss, test_acc = self.eval(test_loader)

            # モデルの保存, 上書き
            torch.save(self.model.to('cpu').state_dict(), os.path.join(self.pth_dir, self.pth_name))
            self.model.to(self.device)

            # csvへの出力, 上書き
            record = [[train_loss, train_acc, train_time, test_loss, test_acc]]
            print(record)
            df = pd.DataFrame(record, columns=None, index=None)
            if os.path.isfile(os.path.join(self.csv_dir, self.csv_name)):
                df.to_csv(os.path.join(self.csv_dir, self.csv_name), mode='a', header=None, index=None)
            else:
                header = ['train_loss', 'train_acc', 'train_time', 'test_loss', 'test_acc']
                df.to_csv(os.path.join(self.csv_dir, self.csv_name), mode='a', header=header, index=None)

            if self.scheduler:
                self.scheduler.step()

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

