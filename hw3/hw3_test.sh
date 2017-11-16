#!/bin/bash
wget -O save/model.h5 "https://www.dropbox.com/s/20cki2hw8qbz6mm/model.h5?dl=0"
python3 test.py --filename "$1" --output "$2"
