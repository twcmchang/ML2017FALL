#!/bin/bash
wget -O w2v.zip "https://www.dropbox.com/s/jjt31y0d5cyofp2/w2v.zip?dl=0"
unzip w2v.zip -d w2v/
python3 train.py --train_file "$1" --unlabeled_file "$2" --w2v_vocab w2v/vocab.pkl --w2v_weight w2v/w2v_weight.pkl --save_dir save
