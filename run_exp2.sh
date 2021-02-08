#!/usr/bin/env bash

! ssh -t u60699 'cd attention-is-all-you-need-pytorch; exec $SHELL'

! python preprocess.py -lang_src pt -lang_trg spt -share_vocab -save_data m30k_deen_shr.pkl

! python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing \
         -save_model trained -b 64 -warmup 12800 -n_layers 8 -n_head 10 -epoch 30


! python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
