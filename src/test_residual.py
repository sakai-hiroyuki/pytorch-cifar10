from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchsummary import summary

from experiment import ExperimentCIFAR10
from models import resnet20_cifar10


if __name__ == '__main__':
    model = resnet20_cifar10()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    print(optimizer)
    scheduler = MultiStepLR(optimizer, milestones=[32000, 480000], gamma=0.1)
    summary(model, [3, 32, 32])

    experiment = ExperimentCIFAR10(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        max_epoch = 300,
        batch_size = 128,
        cutout = False,
        csv_dir = '.',
        csv_name = 'test_residual.csv',
        pth_dir = '.',
        pth_name = 'test_residual.pth',
        use_tqdm=True,
    )

    experiment()
