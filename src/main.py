from argparse import ArgumentParser

from torch import nn

from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR

from efficientnet_pytorch import EfficientNet
from adabelief_pytorch import AdaBelief
from torchsummary import summary

from experiment import ExperimentCIFAR10
from models import resnet20_cifar10


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--max_epoch', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-o', '--optimizer', type=str, default='momentum')
    parser.add_argument('-m', '--model', type=str, default='resnet20-cifar10')
    parser.add_argument('-s', '--scheduler', type=int, default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-csv', '--csv_dir', type=str, default='results/csv')
    parser.add_argument('-pth', '--pth_dir', type=str, default='results/pth')
    args = parser.parse_args()

    optimizer_dict = {
        'sgd': (SGD, {}),
        'momentum': (SGD, {'momentum': 0.9}),
        'rmsprop': (RMSprop, {}),
        'adam': (Adam, {}),
        'amsgrad': (Adam, {'amsgrad': True}),
        'adabelief': (AdaBelief, {}),
    }

    if args.model == 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, 10)
    elif args.model == 'resnet20-cifar10':
        model = resnet20_cifar10()
    else:
        raise ValueError(f'Invaild model: {args.model}')
    summary(model, [3, 32, 32])

    optimizer = optimizer_dict[args.optimizer][0](
        model.parameters(),
        lr = args.learning_rate,
        weight_decay = args.weight_decay,
        **optimizer_dict[args.optimizer][1]
    )
    print(optimizer)

    if args.scheduler == 0:
        scheduler = None
    elif args.scheduler == 1:
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    else:
        raise ValueError(f'Invaild scheduler: {args.scheduler}')
    
    print(f'lr_scheduler: {not args.scheduler == 0}')
    
    csv_name = f'{args.optimizer}.csv'
    pth_name = f'{args.optimizer}.pth'

    experiment = ExperimentCIFAR10(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        max_epoch = args.max_epoch,
        batch_size = args.batch_size,
        csv_dir = args.csv_dir,
        csv_name = csv_name,
        pth_dir = args.pth_dir,
        pth_name = pth_name,
        use_tqdm=True,
    )

    experiment()
