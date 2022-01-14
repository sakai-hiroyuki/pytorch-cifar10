import os
import datetime
import pandas as pd
from tqdm import tqdm
from time import time

import torch
from torch import nn
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from efficientnet_pytorch import EfficientNet
from torchsummary import summary

__all__ = ['experiment']


def prepare_data(data_dir, batch_size=256):
    train_data = CIFAR10(data_dir, train=True, download=True, transform=ToTensor())
    test_data = CIFAR10(data_dir, train=False, download=False, transform=ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def experiment(
    model,
    optimizer,
    max_epoch=100,
    data_dir='./data',
    csv_dir='results/csv',
    csv_name=None,
    prm_dir='results/prm',
    prm_name=None
):
    if csv_name is None:
        csv_name = f'{_now2str()}.csv'
    if prm_name is None:
        prm_name = f'{_now2str()}.prm'

    # モデルが存在するならロードする.
    if os.path.isfile(os.path.join(prm_dir, prm_name)):
        print(f'{os.path.join(prm_dir, prm_name)} already exists.')
        model.load_state_dict(torch.load(os.path.join(prm_dir, prm_name)))
    
    device = 'cuda:0' if is_available() else 'cpu'
    model = model.to(device)

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    if not os.path.isdir(csv_dir):
        os.makedirs(csv_dir)
    if not os.path.isdir(prm_dir):
        os.makedirs(prm_dir)

    loaders = prepare_data(data_dir)
    train_loader = loaders[0]
    test_loader = loaders[1]

    for _ in tqdm(range(max_epoch)):
        train_loss, train_acc, train_time = train(model, optimizer, train_loader, device)
        test_loss, test_acc = eval(model, test_loader, device)

        # モデルの保存, 上書き
        torch.save(model.to('cpu').state_dict(), os.path.join(prm_dir, prm_name))
        model.to(device)

        # csvへの出力, 上書き
        record = [[train_loss, train_acc, train_time, test_loss, test_acc]]
        df = pd.DataFrame(record, columns=None, index=None)
        if os.path.isfile(os.path.join(csv_dir, csv_name)):
            df.to_csv(os.path.join(csv_dir, csv_name), mode='a', header=None, index=None)
        else:
            header = ['train_loss', 'train_acc', 'train_time', 'test_loss', 'test_acc']
            df.to_csv(os.path.join(csv_dir, csv_name), mode='a', header=header, index=None)


def train(model, optimizer, train_loader, device, use_tqdm=True):
    model.train()
    criterion = nn.CrossEntropyLoss()
    i = 0
    total = 0
    correct = 0
    running_loss = 0
    tic = time()

    if use_tqdm:
        train_loader = tqdm(train_loader)

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        running_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i += 1

    train_loss = running_loss / i
    train_acc = correct / total
    train_time = time() - tic
    return train_loss, train_acc, train_time


def eval(model, test_loader, device):
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    i = 0    # ステップ数
    total = 0    # 全てのテストデータの数
    correct = 0    # 正しく分類されたテストデータの数
    running_loss = 0.0  # 予測損失の合計

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
                
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i += 1

    test_loss = running_loss / i
    test_acc = correct / total
    return test_loss, test_acc


def _now2str():
    tzinfo = datetime.timezone(datetime.timedelta(hours=9))
    dt_now = datetime.datetime.now(tz=tzinfo)
    _year = str(dt_now.year).zfill(2)
    _month = str(dt_now.month).zfill(2)
    _day = str(dt_now.day).zfill(2)
    _hour = str(dt_now.hour).zfill(2)
    _minute = str(dt_now.minute).zfill(2)
    _second = str(dt_now.second).zfill(2)
    _now = f'{_year}-{_month}-{_day}-{_hour}:{_minute}:{_second}'
    return _now
