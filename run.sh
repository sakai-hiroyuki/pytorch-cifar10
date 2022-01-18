#!/bin/sh

python src/main.py --csv_dir 'results/csv' \
                   --pt_dir 'results/pt' \
                   --max_epoch 100 \
                   --batch_size 256 \
                   --cutout \
                   --optimizer 'sgd'
