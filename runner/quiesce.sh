#!/bin/bash

# Disable Turbo Boost
# On AMD
sudo sh -c "echo 0 > /sys/devices/system/cpu/cpufreq/boost"
# On Intel
# sudo sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"

# We may need to do that for all cpus.
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
