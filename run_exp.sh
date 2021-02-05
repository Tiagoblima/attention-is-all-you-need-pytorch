#!/usr/bin/env bash

! pip install spacy==2.2.2
! python -m spacy download pt
! pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html

! wget https://github.com/Tiagoblima/ts-corpus-mt/raw/main/simp_mt.zip
! unzip simp_mt.zip

! python preprocess.py -lang_src pt -lang_trg spt -share_vocab -save_data m30k_deen_shr.pkl

! python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing \
         -save_model trained -b 128 -warmup 12800 -epoch 30


! python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
