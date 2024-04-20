#!/usr/bin/zsh

find . -name 'bench.ipynb' -exec jupyter nbconvert --to python {} \; 
for f in $(find . -name 'bench.ipynb'); do cp $f $(dirname $f)/bench-dias.ipynb; cp $f $(dirname $f)/bench-modin.ipynb; done