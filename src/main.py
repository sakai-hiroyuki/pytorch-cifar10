from argparse import ArgumentParser

from torch import nn
from torch.optim import Adam, SGD

from efficientnet_pytorch import EfficientNet
from adabelief_pytorch import AdaBelief
from torchsummary import summary

from experiment import experiment


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='results/csv/efficientnet-b0')
    parser.add_argument('--prm_dir', type=str, default='results/prm/efficientnet-b0')
    args = parser.parse_args()
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
        optimizer = opt_dict[opt_name](model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        summary(model, [3, 32, 32])

        experiment(
            model = model,
            optimizer = optimizer,
            max_epoch = 200,
            csv_dir = csv_dir,
            csv_name = f'{opt_name}.csv',
            prm_dir = prm_dir,
            prm_name = f'{opt_name}.prm',
        )
