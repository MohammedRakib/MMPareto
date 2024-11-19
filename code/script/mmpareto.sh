#!/bin/bash
python MMPareto.py \
--dataset CREMAD \
--model MMPareto \
--gpu_ids 1 \
--n_classes 6 \
--batch_size  16 \
--epochs 100 \
--learning_rate 0.001 \
--lr_decay_step 70 \
--lr_decay_ratio 0.1 \
--train \
| tee log_print/MMPareto-CREMAD.log


# python MMPareto.py \
# --dataset AVMNIST \
# --model MMPareto \
# --gpu_ids 0 \
# --n_classes 10 \
# --batch_size  16 \
# --epochs 100 \
# --learning_rate 0.001 \
# --lr_decay_step 70 \
# --lr_decay_ratio 0.1 \
# --train \
# | tee log_print/MMPareto-AVMNIST.log


# python MMPareto.py \
# --dataset VGGSound \
# --model MMPareto \
# --gpu_ids 0 \
# --n_classes 309 \
# --batch_size  16 \
# --epochs 100 \
# --learning_rate 0.001 \
# --lr_decay_step 70 \
# --lr_decay_ratio 0.1 \
# --train \
# | tee log_print/MMPareto-VGGSound.log