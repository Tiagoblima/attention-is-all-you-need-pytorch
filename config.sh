#!/usr/bin/env bash

#! git clone https://github.com/Tiagoblima/attention-is-all-you-need-pytorch.git

#! ssh -t u60699 'cd attention-is-all-you-need-pytorch; exec $SHELL'

! pwd

! pip install spacy==2.2.2
! python -m spacy download pt
! pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html

! wget https://github.com/Tiagoblima/ts-corpus-mt/raw/main/simp_mt.zip
! unzip simp_mt.zip

! python preprocess.py -lang_src pt -lang_trg spt -share_vocab -save_data m30k_deen_shr.pkl

! wget -O checkpoint_no_cuda.zip https://www.dropbox.com/s/fqxl5kw83j5s7xr/checkpoint_no_cuda.zip?dl=1

! unzip checkpoint_no_cuda.zip
