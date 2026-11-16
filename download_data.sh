#!/usr/bin/env bash
# Fetch both public datasets into ./data (about 15 MB).
set -euo pipefail
mkdir -p data/picsdb data/spc2015

# PhysioNet PICSDB v1.0.0: R-peak (.qrsc) and bradycardia (.atr) annotations only
P=https://physionet.org/files/picsdb/1.0.0
for i in $(seq 1 10); do
  for e in hea qrsc atr; do curl -sfo "data/picsdb/infant${i}_ecg.$e" "$P/infant${i}_ecg.$e"; done
done

# IEEE SPC 2015 training set (12 recordings), via a public GitHub mirror
S=https://raw.githubusercontent.com/MartinBrans/PPG-Heart-Rate-Tracking/HEAD/datasets/troika/training_data
for f in 01_TYPE01 02_TYPE02 03_TYPE02 04_TYPE01 04_TYPE02 05_TYPE02 06_TYPE02 07_TYPE02 08_TYPE02 10_TYPE02 11_TYPE02 12_TYPE02; do
  curl -sfo "data/spc2015/DATA_$f.mat" "$S/DATA_$f.mat"
  curl -sfo "data/spc2015/REF_$f.mat" "$S/REF_$f.mat"
done
echo "done: $(ls data/picsdb | wc -l) PICSDB files, $(ls data/spc2015 | wc -l) SPC files"
