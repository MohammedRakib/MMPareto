#!/bin/bash
# python MMPareto.py \
# --dataset CREMAD \
# --model MMPareto \
# --gpu_ids 2 \
# --n_classes 6 \
# --batch_size  64 \
# --epochs 50 \
# --learning_rate 0.002 \
# --lr_decay_step 30 \
# --lr_decay_ratio 0.1 \
# --train \
# | tee log_print/MMPareto-CREMAD.log


# python MMPareto.py \
# --dataset AVMNIST \
# --model MMPareto \
# --gpu_ids 2 \
# --n_classes 10 \
# --batch_size  64 \
# --epochs 50 \
# --learning_rate 0.002 \
# --lr_decay_step 30 \
# --lr_decay_ratio 0.1 \
# --train \
# | tee log_print/MMPareto-AVMNIST.log


python MMPareto.py \
--dataset VGGSound \
--model MMPareto \
--gpu_ids 2 \
--n_classes 309 \
--batch_size  64 \
--epochs 10 \
--learning_rate 0.002 \
--lr_decay_step 30 \
--lr_decay_ratio 0.1 \
--train \
| tee log_print/MMPareto-VGGSound.log