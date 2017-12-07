#!/bin/bash
wget -O model.zip "https://www.dropbox.com/s/wkhhzx9o8eed79l/save.zip?dl=0"
unzip model.zip -d save/ 
python3 test.py --test_file "$1" --output "$2" --save_dir save/ --model save/model.h5
