curl -O -L 'https://www.dropbox.com/scl/fi/h3au34o4p4h2qbf5b6kq8/data.zip?rlkey=4qyh1d00epi57ub5yrj5drv64&dl=0'
unzip data.zip
python preprocessing.py
python evaluate.py