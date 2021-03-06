#!/usr/bin/env bash

#! git clone https://github.com/Tiagoblima/attention-is-all-you-need-pytorch.git

! ssh -t u60699 'cd attention-is-all-you-need-pytorch; exec $SHELL'

! ssh u63074@login-2

cd  attention-is-all-you-need-pytorch || exit

! pwd

#! pip install spacy==2.2.2
! pip install -r requirements.txt
! python -m spacy download pt
#! pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html

! wget -O glove_s300.zip https://www.dropbox.com/s/s74ovzynh5jbccz/glove_s300.zip?dl=1
! unzip glove_s300.zip

! git clone https://github.com/Tiagoblima/ts-corpus-mt.git
! python preprocess.py -share_vocab -save_data m30k_deen_shr.pkl
! mkdir "checkpoint"
exit 0
