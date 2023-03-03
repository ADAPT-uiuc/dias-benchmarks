#!/bin/bash

./mult_runs.sh --modin 12 --less_replication
./mult_runs.sh --modin 8 --less_replication
./mult_runs.sh --modin 4 --less_replication
./mult_runs.sh --less_replication
./mult_runs.sh
./mult_runs.sh --rewriter --less_replication
./mult_runs.sh --rewriter --no_sliced_exec
./mult_runs.sh --rewriter
./mult_runs.sh --rewriter --rewr_stats
rm -rf stats