#!/bin/bash

# Download NYC Taxi
wget https://dask-data.s3.amazonaws.com/nyc-taxi/2015/yellow_tripdata_2015-01.csv

# Download Iris
python3 download_iris.py
