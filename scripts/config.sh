#!/usr/bin/env bash

#! git clone https://github.com/Tiagoblima/attention-is-all-you-need-pytorch.git

#! ssh -t u60699 'cd attention-is-all-you-need-pytorch; exec $SHELL'

#ssh u63074@login-2

#cd  attention-is-all-you-need-pytorch || exit

! pwd

#! pip install spacy==2.2.2
! pip install -r requirements.txt
! python -m spacy download pt
#! pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html

! wget https://github.com/Tiagoblima/ts-corpus-mt/raw/main/simp_mt.zip
! unzip simp_mt.zip

! wget -O weights.txt.npy https://www.dropbox.com/s/xjflhkl0bs4jut8/weights.txt.npy?dl=1

! python preprocess.py -share_vocab -save_data m30k_deen_shr.pkl

! wget -O checkpoint_no_cuda.zip https://www.dropbox.com/s/fqxl5kw83j5s7xr/checkpoint_no_cuda.zip?dl=1

! unzip checkpoint_no_cuda.zip

exit 0
