
import re
import json
import sys
import subprocess
import os
from enum import Enum
import glob

import bench_utils

def import_pandas_alt(source_cells, num_cpus, alt):
  modin_import = \
f"""
import os
os.environ["MODIN_ENGINE"] = "ray"
import ray
os.environ['MODIN_CPUS'] = "{num_cpus}"
ray.init(num_cpus={num_cpus}, runtime_env={{'env_vars': {{'__MODIN_AUTOIMPORT_PANDAS__': '1'}}}})
import modin.pandas as pd
"""

  # TODO: Specify number of cores. We should be able to do sth like:
  # from pyspark import SparkConf, SparkContext
  # conf = SparkConf()
  # conf.get('spark.executor.cores', '4')
  # SparkContext(conf=conf)
  koalas_import = \
"""
import collections.abc
# Koalas needs these mappings
collections.Iterable = collections.abc.Iterable
collections.Callable = collections.abc.Callable
import databricks.koalas as pd

# It seems you need to set this option for performance reasons.
# See: https://github.com/databricks/koalas/issues/1769 (it seems the issue is not only related to apply())
pd.set_option('compute.default_index_type', 'distributed')

# We need this option because e.g., this:
#   train_df['total_score'] = train_df[LABEL_COLUMNS].sum(axis=1)
# doesn't run with Koalas.
pd.options.compute.ops_on_diff_frames = True
"""

  import_stmt = None
  if alt == "modin":
    import_stmt = modin_import
  elif alt == "koalas":
    import_stmt = koalas_import
  else:
    assert 0

  import_split = import_stmt.splitlines(keepends=True)

  new_cells = []
  # Search for "import pandas as pd" and replace it.
  for cell in source_cells:
    pandas_import = "import pandas as pd\n"
    if pandas_import in cell:
      cell_split = cell.splitlines(keepends=True)
      import_idx = cell_split.index(pandas_import)
      cell_split[import_idx:import_idx+1] = import_split
      new_cell = ''.join(cell_split)
    else:
      new_cell = cell
    # END IF #
    new_cells.append(new_cell)
  # END FOR #
  return new_cells

def run_nb_file(nb_path, args):
  source_cells = bench_utils.open_and_get_source_cells(nb_path)

  if args.alt is not None:
    source_cells = import_pandas_alt(source_cells, args.cores, args.alt)

  # Don't do the following. You'll mess the cell index (i.e., we won't know that it is the nth cell)
  # source_cells = [s for s in source_cells if s.strip() != ""]

  src_dir = os.path.dirname(nb_path)

  def run_config(source_cells, error_file, times_file, 
                mem_usg_file, args):
    config = dict()
    config['src_dir'] = src_dir
    config['cells'] = source_cells
    config['error_file'] = error_file
    config['output_times_json'] = times_file
    config['cores'] = args.cores
    config['less_replication'] = args.less_replication
    config['measure_modin_mem'] = args.measure_mem and args.alt == "modin"

    config_filename = 'run_config.json'
    f = open(config_filename, 'w')
    json.dump(config, f, indent=2)
    f.close()

    # We measure memory usage with GNU time -v. We will only take that into account later if Modin
    # is not enabled, because this is unreliable for Modin.
    if args.measure_mem:
      res = subprocess.run(["/usr/bin/time", "-v", "-o", mem_usg_file, "ipython", "log_times.py", f"{config_filename}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
      res = subprocess.run(["ipython", "log_times.py", f"{config_filename}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
  succ, res = run_config(source_cells, err_file, times_file,
                         mem_usg_file, args)
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

  if args.measure_mem and args.alt is None:
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

def run_nb_paper(nb_dir, args):
  nb_file = "bench.ipynb"
  nb_path = "/".join((nb_dir, "src", nb_file))

  return run_nb_file(nb_path, args)