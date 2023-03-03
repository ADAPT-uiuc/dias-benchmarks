#!/bin/bash

counter=0
# This script is supposed to be killed externally. But as a backup, we can create
# this file to kill it.
while ! [ -f "kill_ray_sampler.f" ]; do
  ray memory --stats-only > ray_stats_${counter}.txt
  counter=$(($counter + 1))
  # TODO: `ray memory` has huge latency. On the one hand, we don't want
  # to call it every often if it adds a lot of overhead. OTOH, if we don't
  # call it often, we might miss a max value (i.e., the memory rises and drops
  # in a time interval and we didn't call it at the peak). 2s seems a good tradeoff.
  sleep 2
done