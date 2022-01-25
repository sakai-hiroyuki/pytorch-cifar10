#!/bin/sh

python src/main.py -o 'momentum' \
                    -m 'resnet20-cifar10' \
                    -e 100 \
                    -b 1024 \
                    -csv 'results/csv' \
                    -pth 'results/pth' \
                    --weight_decay 1e-4
