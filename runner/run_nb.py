
import re
import json
import sys
import subprocess
import os
from enum import Enum
import glob

import bench_utils

def import_pandas_alt(source_cells, num_cpus, alt):
  def modin_import(name_for_modin_pandas):
    return \
f"""
import os
os.environ["MODIN_ENGINE"] = "ray"
import ray
os.environ['MODIN_CPUS'] = "{num_cpus}"
ray.init(num_cpus={num_cpus}, runtime_env={{'env_vars': {{'__MODIN_AUTOIMPORT_PANDAS__': '1'}}}})
import modin.pandas as {name_for_modin_pandas}
"""

  std_modin_import = modin_import("pd")

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

  analytical_import = \
f"""
import pandas as pd
import functools
{modin_import("modin_pd")}

os.environ["USE_MODIN"]="True"

from typing import Dict, List, Optional, Tuple, Any, Literal, Callable
from pandas._typing import AggFuncType, Axis, IndexLabel, Suffixes

_CS598_save_DataFrame_apply = pd.DataFrame.apply
_CS598_save_DataFrame_merge = pd.DataFrame.merge
_CS598_save_Series_apply = pd.Series.apply
_CS598_save_to_datetime = pd.to_datetime

def _CS598_Series_apply(self, func: AggFuncType, convert_dtype: bool = True,
                        args: tuple[Any, ...] = (), **kwargs):
  default_call = functools.partial(_CS598_save_Series_apply, self, func, convert_dtype, args, **kwargs)
  assert isinstance(self, pd.Series)
  
  if os.environ["USE_MODIN"] == "True":
    modin_ser = modin_pd.Series(self)
    # TODO: We need a generic way to deal with this.
    # Problem: pd.numeric is passed which doesn't work on a modin dataframe.
    if func == pd.to_numeric:
      assert false
      func = modin_pd.to_numeric
    modin_res = modin_ser.apply(func, convert_dtype, args, **kwargs)
    return modin_res._to_pandas()
  return default_call()

def _CS598_DataFrame_apply(
    self,
    func: AggFuncType,
    axis: Axis = 0,
    raw: bool = False,
    result_type: Literal["expand", "reduce", "broadcast"] | None = None,
    args=(),
    **kwargs,
):
  default_call = functools.partial(_CS598_save_DataFrame_apply, self, func, axis, raw, result_type, args, **kwargs)
  assert isinstance(self, pd.DataFrame)
  
  if os.environ["USE_MODIN"] == "True":
    modin_df = modin_pd.DataFrame(self)
    modin_res = modin_df.apply(func, axis, raw, result_type, args, **kwargs)
    return modin_res._to_pandas()
  return default_call()

def _CS598_DataFrame_merge(
    self,
    right: pd.DataFrame | pd.Series,
    how: str = "inner",
    on: IndexLabel | None = None,
    left_on: IndexLabel | None = None,
    right_on: IndexLabel | None = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Suffixes = ("_x", "_y"),
    copy: bool = True,
    indicator: bool = False,
    validate: str | None = None,
) -> pd.DataFrame:
  assert isinstance(self, pd.DataFrame)        
  default_call = functools.partial(_CS598_save_DataFrame_merge, self, right, how, on, 
      left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  
  if os.environ["USE_MODIN"] == "True":
    modin_df = modin_pd.DataFrame(self)
    modin_res = modin_df.merge(right, how, on, 
      left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    return modin_res._to_pandas()
  return default_call()


def _CS598_to_datetime(
    arg,
    errors = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool | None = None,
    format: str | None = None,
    exact: bool = True,
    unit: str | None = None,
    infer_datetime_format: bool = False,
    origin="unix",
    cache: bool = True,
):
  default_call = functools.partial(_CS598_save_to_datetime, arg, errors, dayfirst,
      yearfirst, utc, format, exact, unit,infer_datetime_format, origin, cache)
  
  if os.environ["USE_MODIN"] == "True":
    if type(arg) == pd.DataFrame:
      arg = modin_pd.DataFrame(arg)
    elif type(arg) == pd.Series:
      arg = modin_pd.Series(arg)

    modin_res = modin_pd.to_datetime(arg, errors, dayfirst,
      yearfirst, utc, format, exact, unit,infer_datetime_format, origin, cache)
    return modin_res._to_pandas()
  return default_call()

assert pd.Series.apply != _CS598_Series_apply
assert pd.DataFrame.apply != _CS598_DataFrame_apply
assert pd.DataFrame.merge != _CS598_DataFrame_merge
assert pd.to_datetime != _CS598_to_datetime
# Overwriting is not trivial. Thanks to:
# https://github.com/lux-org/lux/blob/550a2eca90b26c944ebe8600df7a51907bc851be/lux/core/__init__.py#L27
pd.Series.apply = pd.core.frame.Series.apply = pd.core.series.Series.apply = _CS598_Series_apply
pd.DataFrame.apply = pd.core.frame.DataFrame.apply = _CS598_DataFrame_apply
pd.DataFrame.merge = pd.core.frame.DataFrame.merge = _CS598_save_DataFrame_merge
pd.to_datetime = _CS598_to_datetime
"""

  import_stmt = None
  if alt == "modin":
    import_stmt = std_modin_import
  elif alt == "koalas":
    import_stmt = koalas_import
  elif alt == "analytical":
    import_stmt = analytical_import
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

def modify_dataset_size(source_cells, dataset_size):
  search_size_stmt = "intended_df_size_in_MB = 256\n"
  updated_size_stmt = ""
  if dataset_size is not None:
    updated_size_stmt = "intended_df_size_in_MB = " + str(dataset_size) + "\n"
  
  new_cells = []
  for cell in source_cells:
    if search_size_stmt in cell:
      cell_split = cell.splitlines(keepends=True)
      if dataset_size is None:
        # do not do anything related to dataset size
        # last line just prints info
        new_cell = ''.join(cell_split[-1])
      else:
        size_stmt_idx = cell_split.index(search_size_stmt)
        # replace the statement denoting dataset size only
        cell_split[size_stmt_idx] = updated_size_stmt
        new_cell = ''.join(cell_split)
      #print(new_cell)
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

  source_cells = modify_dataset_size(source_cells, args.dataset_size)

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
