#!/usr/bin/env bash

#! ssh -t u60699 'cd attention-is-all-you-need-pytorch; exec $SHELL'

ssh u60699@s001-n004

cd  attention-is-all-you-need-pytorch || exit

! python train.py -data_pkl m30k_deen_shr.pkl -log exp_hist -embs_share_weight -proj_share_weight -label_smoothing \
         -save_model exp -b 128 -warmup 12800 -epoch 20 -no_cuda


! python translate.py -data_pkl m30k_deen_shr.pkl -model exp.chkpt -output prediction_exp.txt
