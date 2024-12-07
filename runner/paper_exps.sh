#!/bin/bash

if [[ ! -v DIAS_ROOT ]]; then
  echo "[ERROR]: DIAS_ROOT is not set. Please set it to the root of the Dias repo."
  exit 1
fi

# Vanilla pandas runs
#
# These two are pretty fast, so for folks trying to reproduce these,
# it's better to have them first so that if they get errors, they
# get them faster.
./mult_runs.sh
./mult_runs.sh --less_replication

# Dias runs
#
# These are the fastest, but we have the dependency on Dias, so they're
# the next step of making sure there are no errors.
./mult_runs.sh --rewriter
./mult_runs.sh --rewriter --less_replication

# Modin time and memory measurements for 4, 8 and 12 cores.
./mult_runs.sh --modin 4 --less_replication
./mult_runs.sh --modin 4 --less_replication --measure_modin_mem
./mult_runs.sh --modin 8 --less_replication
./mult_runs.sh --modin 8 --less_replication --measure_modin_mem
./mult_runs.sh --modin 12 --less_replication
./mult_runs.sh --modin 12 --less_replication --measure_modin_mem

rm -rf stats
