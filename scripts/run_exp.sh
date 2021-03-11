#!/usr/bin/env bash

#! ssh -t u60699 'cd attention-is-all-you-need-pytorch; exec $SHELL'

ssh u63074@login-2

cd  attention-is-all-you-need-pytorch || exit

! python train.py -data_pkl m30k_deen_shr.pkl -log logs/exp1_hist -embs_share_weight -proj_share_weight -label_smoothing \
         -save_model checkpoint/exp1 -b 32 -warmup 12800 -epoch 10


! python translate.py -data_pkl m30k_deen_shr.pkl -model checkpoint/exp1.chkpt -output prediction_exp1.txt

exit 0
