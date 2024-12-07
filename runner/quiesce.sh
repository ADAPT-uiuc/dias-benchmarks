#!/bin/bash

# Disable Turbo Boost
if grep -q "AuthenticAMD" /proc/cpuinfo; then
  # On AMD
  echo "AMD"
  sudo sh -c "echo 0 > /sys/devices/system/cpu/cpufreq/boost"
elif grep -q "GenuineIntel" /proc/cpuinfo; then
  # On Intel
  echo "Intel"
  sudo sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"
else
    echo "COULD NOT IDENTIFY THE CPU!"
fi

# Disable dynamic frequency scaling
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
