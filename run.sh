#!/bin/sh

python src/main.py -o 'momentum' \
                    -m 'resnet20-cifar10' \
                    -s 1 \
                    -e 300 \
                    -b 1024 \
                    -lr 0.1 \
                    -csv 'results/csv' \
                    -pth 'results/pth' \
                    --weight_decay 1e-4
