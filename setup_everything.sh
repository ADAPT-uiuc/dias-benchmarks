#!/bin/bash

sudo apt-get update
sudo apt install python3.10 python3-pip python3.10-venv -y

python3 -m venv env
source env/bin/activate

export DIAS_ROOT=$(pwd)/../dias

cd runner
pip install -r requirements.txt
# It's now in dias-benchmarks root
cd ../

wget https://uofi.box.com/shared/static/9r1fgjdpoz113ed2al7k1biwxgnn9fpa -O dias_datasets.zip
# This should create a directory named dias-datasets
unzip dias_datasets.zip

cd dias-datasets
./copier.sh ../notebooks
cd ../
./verify_datasets_are_copied.sh

cd runner
./quiesce.sh
./pre_run.sh