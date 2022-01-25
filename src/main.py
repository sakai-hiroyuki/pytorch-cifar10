from argparse import ArgumentParser

from torch import nn

# モデル関連
from models import (
    resnet20_cifar10,
    resnet32_cifar10,
    resnet44_cifar10,
    resnet56_cifar10,
    resnet110_cifar10,
)
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

# 最適化アルゴリズム関連
from torch.optim import Adam, SGD, RMSprop
from adabelief_pytorch import AdaBelief

# 学習率関連
from torch.optim.lr_scheduler import StepLR

from experiment import ExperimentCIFAR10


def str2model(s):
    if s == 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, 10)
        return model
    if s == 'resnet20-cifar10':
        return resnet20_cifar10()
    if s == 'resnet32-cifar10':
        return resnet32_cifar10()
    if s == 'resnet44-cifar10':
        return resnet44_cifar10()
    if s == 'resnet56-cifar10':
        return resnet56_cifar10()
    if s == 'resnet110-cifar10':
        return resnet110_cifar10()

    raise ValueError(f'Invaild model: {s}')


def int2scheduler(n):
    if n == 0:
        return None
    if n == 1:
        return StepLR(optimizer, step_size=100, gamma=0.1)
    raise ValueError(f'Invaild scheduler: {args.scheduler}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e',   '--max_epoch', type=int, default=300)
    parser.add_argument('-b',   '--batch_size', type=int, default=1024)
    parser.add_argument('-o',   '--optimizer', type=str, default='momentum')
    parser.add_argument('-m',   '--model', type=str, default='resnet20-cifar10')
    parser.add_argument('-s',   '--scheduler', type=int, default=1)
    parser.add_argument('-lr',  '--learning_rate', type=float, default=1e-1)
    parser.add_argument('-wd',  '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-csv', '--csv_dir', type=str, default='results/csv')
    parser.add_argument('-pth', '--pth_dir', type=str, default='results/pth')
    args = parser.parse_args()

    optimizer_dict = {
        'sgd'      : (SGD,       {}),
        'momentum' : (SGD,       {'momentum': 0.9}),
        'rmsprop'  : (RMSprop,   {}),
        'adam'     : (Adam,      {}),
        'amsgrad'  : (Adam,      {'amsgrad': True}),
        'adabelief': (AdaBelief, {}),
    }

    model = str2model(args.model)
    summary(model, [3, 32, 32])

    optimizer = optimizer_dict[args.optimizer][0](
        model.parameters(),
        lr = args.learning_rate,
        weight_decay = args.weight_decay,
        **optimizer_dict[args.optimizer][1]
    )
    print(optimizer)

    scheduler = int2scheduler(args.scheduler)
    print(f'lr_scheduler: {scheduler is not None}')
    
    csv_name = f'{args.optimizer}.csv'
    pth_name = f'{args.optimizer}.pth'

    experiment = ExperimentCIFAR10(
        model      = model,
        optimizer  = optimizer,
        scheduler  = scheduler,
        max_epoch  = args.max_epoch,
        batch_size = args.batch_size,
        csv_dir    = args.csv_dir,
        csv_name   = csv_name,
        pth_dir    = args.pth_dir,
        pth_name   = pth_name,
        use_tqdm   = True,
    )

    experiment()
