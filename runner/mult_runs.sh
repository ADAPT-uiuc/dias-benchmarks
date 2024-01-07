#!/bin/bash

mkdir -p stats
echo "######### CONFIG: ${@} ############"
for i in {0..9}
do
  echo "------- ITERATION ${i} -----------"
  python -u run_all.py $@ || { echo 'python run_all.py FAILED. Exiting.'; exit 1; }
  mv stats s-${i}
  mkdir stats
done

# Read one of the version files
VERSION=`cat s-0/.version`  
mkdir stats-${VERSION}
mv s-* stats-${VERSION}
