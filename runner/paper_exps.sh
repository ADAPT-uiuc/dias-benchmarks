#!/bin/bash

# Modin time and memory measurements for 4, 8 and 12 cores.
./mult_runs.sh --modin 4 --less_replication
./mult_runs.sh --modin 4 --less_replication --measure_modin_mem
./mult_runs.sh --modin 8 --less_replication
./mult_runs.sh --modin 8 --less_replication --measure_modin_mem
./mult_runs.sh --modin 12 --less_replication
./mult_runs.sh --modin 12 --less_replication --measure_modin_mem

./mult_runs.sh
./mult_runs.sh --less_replication

./mult_runs.sh --rewriter
./mult_runs.sh --rewriter --less_replication
rm -rf stats