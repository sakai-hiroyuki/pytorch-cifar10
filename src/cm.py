from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from efficientnet_pytorch import EfficientNet


def cm(path):
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 10)
    model.load_state_dict(torch.load(path))

    test_data = CIFAR10('./data', train=False, download=False, transform=ToTensor())
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

    y_true = []
    y_pred = []
    for inputs, labels in tqdm(test_loader):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true += labels.to('cpu').detach().numpy().tolist()
        y_pred += predicted.to('cpu').detach().numpy().tolist()

    return confusion_matrix(y_true, y_pred)


if __name__ == '__main__':
    print(cm('results/pth/efficientnet-b0/adam+cutout.pth'))
