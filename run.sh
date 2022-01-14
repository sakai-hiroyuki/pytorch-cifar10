#!/bin/sh

python src/main.py --csv_dir 'results/csv/efficientnet-b0' \
                   --prm_dir 'results/prm/efficientnet-b0' \
                   --max_epoch 100 \
                   --batch_size 256 \
                   --cutout
