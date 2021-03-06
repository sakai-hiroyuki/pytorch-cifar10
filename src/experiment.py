import os
import pandas as pd
from tqdm import tqdm
from time import time

import torch
from torch import nn
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop


class ExperimentCIFAR10:
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(
        self,
        model,
        optimizer,
        scheduler  = None,
        max_epoch  = 100,
        batch_size = 512,
        data_dir   = './data',
        csv_dir    = 'results/csv',
        csv_name   = 'record.csv',
        pth_dir    = 'results/pth',
        pth_name   = 'params.pth',
        use_tqdm   = True
    ) -> None:

        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.max_epoch  = max_epoch
        self.batch_size = batch_size
        self.data_dir   = data_dir
        self.csv_dir    = csv_dir
        self.csv_name   = csv_name
        self.pth_dir    = pth_dir
        self.pth_name   = pth_name
        self.use_tqdm   = use_tqdm

        self.device = 'cuda:0' if is_available() else 'cpu'
        self._max_acc = 0.
    
    def __call__(self):
        self.run()

    def prepare_data(self):
        train_transforms = Compose([
            RandomCrop(32, padding=4, padding_mode='reflect'), 
            RandomHorizontalFlip(), 
            ToTensor(), 
            Normalize(
                mean = (0.4914, 0.4822, 0.4465),
                std = (0.2023, 0.1994, 0.2010),
                inplace = True
            )
        ])
        test_transforms = Compose([
            ToTensor(),
            Normalize(
                mean = (0.4914, 0.4822, 0.4465),
                std = (0.2023, 0.1994, 0.2010)
            )
        ])

        train_data = CIFAR10(self.data_dir, train=True, download=True, transform=train_transforms)        
        test_data = CIFAR10(self.data_dir, train=False, download=False, transform=test_transforms)
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
    
    def run(self):
        self.model.to(self.device)

        # ???????????????????????????????????????????????????.
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
            
            if self.scheduler:
                self.scheduler.step()

            # csv????????????, ?????????
            self.to_csv(train_loss, train_acc, train_time, test_loss, test_acc)

            # ??????????????????, ?????????
            if self._max_acc < test_acc:
                torch.save(self.model.to('cpu').state_dict(), os.path.join(self.pth_dir, self.pth_name))
                self._max_acc = test_acc
                self.model.to(self.device)
        
    def to_csv(self, train_loss, train_acc, train_time, test_loss, test_acc):
        print()
        print(f'train loss: {train_loss}, train acc: {train_acc}')
        print(f'test loss : {test_loss}, test acc : {test_acc}')
        print(f'best: {self._max_acc < test_acc}')

        df = pd.DataFrame([[train_loss, train_acc, train_time, test_loss, test_acc]], columns=None, index=None)
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

    @torch.no_grad()
    def eval(self, test_loader):
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        i = 0               # ???????????????
        total = 0           # ?????????????????????????????????
        correct = 0         # ????????????????????????????????????????????????
        running_loss = 0.0  # ?????????????????????

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
