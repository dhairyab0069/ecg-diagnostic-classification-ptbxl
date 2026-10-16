#!/bin/bash
# Downloads PTB-XL v1.0.3 (about 3 GB) into data/ptb-xl/1.0.3
set -e
mkdir -p data
wget -r -N -c -np -nH --cut-dirs=1 -P data https://physionet.org/files/ptb-xl/1.0.3/
echo "Done. Data is in data/ptb-xl/1.0.3"
