#!/bin/bash
wget -O save.zip "https://www.dropbox.com/s/wye7s3fktdk38jr/save.zip?dl=0"
unzip save.zip -d save/ 
python3 test.py --test_file "$1" --output "$2" --movie_file "$3" --user_file "$4" --save_dir save/
