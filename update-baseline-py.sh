#!/usr/bin/zsh

find . -name 'bench.ipynb' -exec jupyter nbconvert --to python {} \; 