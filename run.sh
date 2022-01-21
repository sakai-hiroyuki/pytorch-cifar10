#!/bin/sh

python src/main.py -o 'momentum' \
                    -e 100 \
                    -b 256 \
                    -csv 'results/csv' -pth 'results/pt' \
                    --cutout \
