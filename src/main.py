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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def prepare_data(data_dir, batch_size=256, cutout=False):
    train_transforms = [ToTensor()]
    if cutout:
        train_transforms.append(Cutout(1, 16))
    train_data = CIFAR10(data_dir, train=True, download=True, transform=Compose(train_transforms))
    test_transforms = [ToTensor()]
    test_data = CIFAR10(data_dir, train=False, download=False, transform=Compose(test_transforms))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run(
    model,
    optimizer,
    max_epoch=100,
    batch_size=256,
    cutout=False,
    data_dir='./data',
    csv_dir='results/csv',
    csv_name=None,
    pt_dir='results/pt',
    pt_name=None
):
    if csv_name is None:
        csv_name = f'record.csv'
    if pt_name is None:
        pt_name = f'model.pt'

    # モデルが存在するならロードする.
    if os.path.isfile(os.path.join(pt_dir, pt_name)):
        print(f'{os.path.join(pt_dir, pt_name)} already exists.')
        model.load_state_dict(torch.load(os.path.join(pt_dir, pt_name)))
    
    device = 'cuda:0' if is_available() else 'cpu'
    model = model.to(device)

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    if not os.path.isdir(csv_dir):
        os.makedirs(csv_dir)
    if not os.path.isdir(pt_dir):
        os.makedirs(pt_dir)

    loaders = prepare_data(data_dir, batch_size=batch_size, cutout=cutout)
    train_loader = loaders[0]
    test_loader = loaders[1]

    for _ in tqdm(range(max_epoch)):
        train_loss, train_acc, train_time = train(model, optimizer, train_loader, device)
        test_loss, test_acc = eval(model, test_loader, device)

        # モデルの保存, 上書き
        torch.save(model.to('cpu').state_dict(), os.path.join(pt_dir, pt_name))
        model.to(device)

        # csvへの出力, 上書き
        record = [[train_loss, train_acc, train_time, test_loss, test_acc]]
        df = pd.DataFrame(record, columns=None, index=None)
        if os.path.isfile(os.path.join(csv_dir, csv_name)):
            df.to_csv(os.path.join(csv_dir, csv_name), mode='a', header=None, index=None)
        else:
            header = ['train_loss', 'train_acc', 'train_time', 'test_loss', 'test_acc']
            df.to_csv(os.path.join(csv_dir, csv_name), mode='a', header=header, index=None)


def train(model, optimizer, train_loader, device, use_tqdm=False):
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--csv_dir', type=str, default='results/csv')
    parser.add_argument('--pt_dir', type=str, default='results/pt')
    parser.add_argument('--optimizer', type=str, default='sgd')
    args = parser.parse_args()

    opt_dict = {
        'adam': Adam,
        'sgd': SGD,
        'adabelief': AdaBelief
    }


    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 10)
    summary(model, [3, 32, 32])

    optimizer = opt_dict[args.optimizer](model.parameters(), lr=1e-3)

    csv_name = f'{args.optimizer}.csv'
    if args.cutout:
        csv_name = csv_name[0:-4] + '+cutout.csv'
 
    pt_name = f'{args.optimizer}.pt'
    if args.cutout:
        pt_name = pt_name[0:-3] + '+cutout.pt'

    run(
        model = model,
        optimizer = optimizer,
        max_epoch = args.max_epoch,
        batch_size = args.batch_size,
        cutout = args.cutout,
        csv_dir = args.csv_dir,
        csv_name = csv_name,
        pt_dir = args.pt_dir,
        pt_name = pt_name,
    )
