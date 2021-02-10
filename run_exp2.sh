#!/usr/bin/env bash

#! ssh -t u60699 'cd attention-is-all-you-need-pytorch; exec $SHELL'

ssh u60699@s001-n004

cd  attention-is-all-you-need-pytorch || exit


! python train.py -data_pkl m30k_deen_shr.pkl -log exp2_hist -embs_share_weight -proj_share_weight -label_smoothing \
         -save_model exp2 -b 64 -warmup 12800 -n_layers 8 -n_head 10 -epoch 20 -no_cuda


! python translate.py -data_pkl m30k_deen_shr.pkl -model exp2.chkpt -output prediction.txt
