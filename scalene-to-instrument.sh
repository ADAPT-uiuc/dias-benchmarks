#!/usr/bin/env zsh

for f in $(cat to-instrument.txt); do
    cd $f
    python3 -m scalene --cli --outfile profile-$(dirname $f).json $f/bench.py
    popd

done