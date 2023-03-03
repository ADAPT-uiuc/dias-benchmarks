
import re
import json
import sys
import subprocess
import os
from enum import Enum
import glob

import bench_utils

def open_and_get_source_cells(nb_path):
  f = open(nb_path)
  nb_as_json = json.load(f)
  f.close()
  return bench_utils.get_nb_source_cells(nb_as_json)

def run_nb(nb_dir, enable_rewriter: bool, modin_cores: int, less_replication: bool, no_sliced_exec: bool, rewr_stats: bool):
  split = nb_dir.split('/')
  kernel_user = split[-2]
  kernel_slug = split[-1]

  nb_file = "bench.ipynb"
  nb_path = "/".join((nb_dir, "src", nb_file))
  source_cells = open_and_get_source_cells(nb_path)

  # Don't do the following. You'll mess the cell index (i.e., we won't know that it is the nth cell)
  # source_cells = [s for s in source_cells if s.strip() != ""]

  src_dir = "/".join((nb_dir, "src"))

  def run_config(add_rewrite, source_cells, error_file, times_file, 
                mem_usg_file, modin_cores, less_replication, no_sliced_exec, rewr_stats):
    config = dict()
    config['src_dir'] = src_dir
    config['rewrite'] = 1 if add_rewrite else 0
    config['cells'] = source_cells
    config['error_file'] = error_file
    config['output_times_json'] = times_file
    config['modin_cores'] = modin_cores
    config['less_replication'] = less_replication
    config['no_sliced_exec'] = no_sliced_exec
    config['rewr_stats'] = rewr_stats

    config_filename = 'run_config.json'
    f = open(config_filename, 'w')
    json.dump(config, f, indent=2)
    f.close()

    if modin_cores == -1:
      # Running without Modin. We can measure memory usage with GNU time -v
      res = subprocess.run(["/usr/bin/time", "-v", "-o", mem_usg_file, "ipython", "log_times.py", f"{config_filename}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      return res.returncode == 0, res
    else:
      # Modin uses Ray. Ray processes don't seem to be suprocesses of the Python instance (Ray probably uses a
      # daemon and when we init an instance, we communicate with that). So, any standard memory profiler like
      # `time` or `mprof` will not measure its memory usage correctly (in fact, it will be off by _a lot_).
      # Fortunately, we can use `ray memory`, a CLI tool. For this, we will launch ray_memory_sampler.sh on the 
      # background (i.e., unblocked), which will keep calling `ray memory` periodically to get the memory usage.

      # Redirect stdout and stderr. It's going to output things for which we don't care.
      ray_sampler = subprocess.Popen("./ray_memory_sampler.sh", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      # Start it blocked, wait for it to finish.
      res = subprocess.run(["ipython", "log_times.py", f"{config_filename}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      # Kill the sampler.
      ray_sampler.kill()
      # It might not be killed immediately, wait.
      ray_sampler.communicate()

      return res.returncode == 0, res

  def load_json(file):
    f = open(file, 'r')
    jd = json.load(f)
    f.close()
    return jd

  pwd = os.getcwd()

  err_file = pwd + '/' + 'error.txt'
  times_file = pwd + '/' + 'times.json'
  mem_usg_file = pwd + '/' + 'mem.txt'
  succ, res = run_config(enable_rewriter, source_cells, err_file, times_file,
                         mem_usg_file, modin_cores, less_replication, no_sliced_exec, rewr_stats)
  the_stdout = res.stdout.decode()
  # We may have an exception which is not denoted as error unfortunately. We have to search the stdout.
  if "Traceback" in the_stdout:
    succ = False
  if not succ:
    print(f"There was an error while running the original {kernel_user}/{kernel_slug}. See {err_file}, stderr.txt and stdout.txt")
    bench_utils.write_to_file("stdout.txt", res.stdout.decode())
    bench_utils.write_to_file("stderr.txt", res.stderr.decode())
    return False
  times = load_json(times_file)
  os.remove(times_file)
  assert len(times['times']) == len(source_cells)
  
  os.remove('run_config.json')

  mem_usage_dict = dict()
  if modin_cores == -1:
    # Parse the results of time -v

    f = open(mem_usg_file, 'r')
    time_v_output = f.read()
    m = re.search("Maximum resident set size \(kbytes\): (\d+)", time_v_output)
    assert m
    in_kbytes = int(m.group(1))
    in_mb = in_kbytes // 1024
    f.close()
    os.remove(mem_usg_file)

    mem_usage_dict['max-mem-in-mb'] = in_mb
    # Neither Pandas nor the rewriter use the disk.
    mem_usage_dict['max-disk-in-mb'] = 0
  else:
    # Parse the results of `ray memory`

    ray_stats_files = glob.glob('ray_stats_*.txt')
    mem = []
    disk = []
    for ray_file in ray_stats_files:
      f = open(ray_file, 'r')
      ray_output = f.read()
      f.close()
      os.remove(ray_file)

      match_mem = re.search("Objects consumed by Ray tasks: (\d+) MiB", ray_output)
      # Some files will contain nothing.
      if match_mem:
        in_mb = int(match_mem.group(1))
        mem.append(in_mb)
      
      match_disk = re.search("Spilled (\d+) MiB", ray_output)
      if match_disk:
        in_mb = int(match_disk.group(1))
        disk.append(in_mb)
    # END OF FOR LOOP #
    # If we have no records for `mem` or `disk`, one of two things have happened:
    # - There was no usage
    # - There was but we didn't catch it with our calls to `ray memory` (possibly
    #   because the notebook was executed too fast and `ray memory` is too slow)
    # The first one cannot be true for memory. Modin must have used some memory. So,
    # if we have no records, we store -1. For disk, we assume there was no usage
    # and we store 0.
    mem_usage_dict['max-mem-in-mb'] = max(mem) if len(mem) > 0 else -1
    mem_usage_dict['max-disk-in-mb'] = max(disk) if len(disk) > 0 else 0
  # END OF mem_usage_dict IF #

  json_d = mem_usage_dict
  json_d['cells'] = []
  for idx in range(len(times['times'])):
    st = dict()
    st['raw'] = source_cells[idx]
    st['wall-time'] = times['times'][idx]
    json_d['cells'].append(st)

  if rewr_stats:
    cells_stats = bench_utils.extract_json_cell_stats(res.stderr.decode())
    assert len(cells_stats) == len(times['times'])
    for idx, st_str in enumerate(cells_stats):
      st = json.loads(st_str)
      json_d['cells'][idx] = json_d['cells'][idx] | st


  f = open('stats.json', 'w')
  json.dump(json_d, f, indent=2)
  f.close()

  return True
