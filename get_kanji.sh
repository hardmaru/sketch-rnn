#!/bin/bash

#fetch and unpack the data
cd data
wget https://github.com/KanjiVG/kanjivg/releases/download/r20150615-2/kanjivg-20150615-2-all.zip
unzip kanjivg-20150615-2-all.zip
# move aside one problem file
mkdir -p rejects
mv kanji/05747-Kaisho.svg rejects
# done, now try: python train.py --dataset_name kanji
