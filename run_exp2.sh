#!/usr/bin/env bash

#! ssh -t u60699 'cd attention-is-all-you-need-pytorch; exec $SHELL'

#ssh u60699@s001-n004

#cd  attention-is-all-you-need-pytorch || exit


! python train.py -data_pkl m30k_deen_shr.pkl -log logs/exp2_hist -embs_share_weight -proj_share_weight -label_smoothing \
         -save_model checkpoint/exp2 -b 64 -warmup 12800 -n_layers 8 -n_head 10 -epoch 1000 -no_cuda


! python translate.py -data_pkl m30k_deen_shr.pkl -model checkpoint/exp2.chkpt -no_cuda -output prediction_exp2.txt
