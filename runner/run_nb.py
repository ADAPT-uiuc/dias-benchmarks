
import re
import json
import sys
import subprocess
import os
from enum import Enum
import glob

import bench_utils

def run_nb_file(nb_path, enable_rewriter: bool, modin_cores: int, less_replication: bool, measure_modin_mem: bool):
  source_cells = bench_utils.open_and_get_source_cells(nb_path)

  # Don't do the following. You'll mess the cell index (i.e., we won't know that it is the nth cell)
  # source_cells = [s for s in source_cells if s.strip() != ""]

  src_dir = os.path.dirname(nb_path)

  def run_config(add_rewrite, source_cells, error_file, times_file, 
                mem_usg_file, modin_cores, less_replication, measure_modin_mem):
    config = dict()
    config['src_dir'] = src_dir
    config['rewrite'] = 1 if add_rewrite else 0
    config['cells'] = source_cells
    config['error_file'] = error_file
    config['output_times_json'] = times_file
    config['modin_cores'] = modin_cores
    config['less_replication'] = less_replication
    config['measure_modin_mem'] = measure_modin_mem

    config_filename = 'run_config.json'
    f = open(config_filename, 'w')
    json.dump(config, f, indent=2)
    f.close()

    # We measure memory usage with GNU time -v. We will only take that into account later if Modin
    # is not enabled, because this is unreliable for Modin.
    res = subprocess.run(["/usr/bin/time", "-v", "-o", mem_usg_file, "ipython", "log_times.py", f"{config_filename}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0, res
  
  # END run_config() #

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
                         mem_usg_file, modin_cores, less_replication, measure_modin_mem)
  the_stdout = res.stdout.decode()
  # We may have an exception which is not denoted as error unfortunately. We have to search the stdout.
  if "Traceback" in the_stdout:
    succ = False
  if not succ:
    print(f"There was an error while running {nb_path}. See {err_file}, stderr.txt and stdout.txt")
    bench_utils.write_to_file("stdout.txt", res.stdout.decode())
    bench_utils.write_to_file("stderr.txt", res.stderr.decode())
    return False
  times = load_json(times_file)
  os.remove(times_file)
  os.remove('run_config.json')

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

    times['max-mem-in-mb'] = in_mb
    # Neither Pandas nor the rewriter use the disk.
    times['max-disk-in-mb'] = 0
  # END if modin_cores == -1 #

  f = open('stats.json', 'w')
  json.dump(times, f, indent=2)
  f.close()

  return True

def run_nb_paper(nb_dir, enable_rewriter: bool, modin_cores: int, less_replication: bool, measure_modin_mem: bool):
  nb_file = "bench.ipynb"
  nb_path = "/".join((nb_dir, "src", nb_file))

  return run_nb_file(nb_path, enable_rewriter, modin_cores, less_replication, measure_modin_mem)