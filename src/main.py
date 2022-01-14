from argparse import ArgumentParser

from torch import nn
from torch.optim import Adam, SGD

from efficientnet_pytorch import EfficientNet
from adabelief_pytorch import AdaBelief
from torchsummary import summary

from experiment import experiment


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--csv_dir', type=str, default='results/csv/efficientnet-b0')
    parser.add_argument('--prm_dir', type=str, default='results/prm/efficientnet-b0')
    args = parser.parse_args()
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    cutout = args.cutout
    csv_dir = args.csv_dir
    prm_dir = args.prm_dir

    opt_dict = {
        'adam': Adam,
        'sgd': SGD,
        'adabelief': AdaBelief
    }

    for opt_name in opt_dict:
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, 10)
        summary(model, [3, 32, 32])

        optimizer = opt_dict[opt_name](model.parameters(), lr=1e-3, betas=(0.9, 0.999))

        if cutout:
            csv_name = f'{opt_name}+cutout.csv'
        else:
            csv_name = f'{opt_name}.csv'
        
        if cutout:
            prm_name = f'{opt_name}+cutout.prm'
        else:
            prm_name = f'{opt_name}.prm'

        experiment(
            model = model,
            optimizer = optimizer,
            max_epoch = max_epoch,
            batch_size=batch_size,
            cutout=cutout,
            csv_dir = csv_dir,
            csv_name = csv_name,
            prm_dir = prm_dir,
            prm_name = prm_name,
        )
