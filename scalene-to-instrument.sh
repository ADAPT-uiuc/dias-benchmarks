#!/usr/bin/env zsh

for f in $(cat to-instrument.txt); do
    pushd $f
    echo "=== RUNNING $f ==="
    python3 -m scalene --cli --json --stacks --outfile profile-$(basename $(pwd)).json $f/bench.py
    if [ $? -ne 0 ]; then
        echo $f >> scalene-failures.txt
    fi
    popd

done