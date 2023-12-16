#!/bin/bash

echo "downloading data"
curl -O -L 'https://www.dropbox.com/scl/fi/h3au34o4p4h2qbf5b6kq8/data.zip?rlkey=4qyh1d00epi57ub5yrj5drv64&dl=0'
unzip -o data.zip
echo "finished downloading data"
echo "preprocessing data"
python preprocessing.py
echo "finished preprocessing data"
echo 'begin evaluation'
python evaluate.py
echo 'finished evaluation'