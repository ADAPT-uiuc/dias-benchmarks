import IPython
import time
from IPython.display import display, Markdown
import pandas as pd
import modin.pandas as modin_pd

@IPython.core.magic.register_line_magic
def include(file):
  f = open(file, 'r')
  contents = f.read()
  f.close()
  IPython.get_ipython().run_cell(contents)

# Execute the line. Return the execution time
@IPython.core.magic.register_line_magic
def my_time(line):
  ip = IPython.get_ipython()
  start = time.perf_counter()
  ip.run_cell("_IREWR_res = (" + line + ")")
  end = time.perf_counter()
  assert "_IREWR_res" in ip.user_ns
  return end - start, ip.user_ns['_IREWR_res']

def print_md(s):
  display(Markdown(s))

def get_two_dfs(path):
  modin_df = modin_pd.read_csv(path)
  pandas_df = pd.read_csv(path)
  return modin_df, pandas_df

# The following two functions are a bit of an overkill

@IPython.core.magic.register_cell_magic
def time_cell(line, cell):
  ip = IPython.get_ipython()
  start = time.perf_counter()
  ip.run_cell(cell)
  end = time.perf_counter()
  ip.user_ns['_TIMED_CELL'] = end-start

# Run the same cell twice. The first time, `ph` points to `first_df`,
# the second to `second_df`. Store the timing results in _TIMED_CELL1 and 2.
@IPython.core.magic.register_cell_magic
def run_twice(line, cell):
  placeholder, first_df, second_df = [s.strip() for s in line.split(',')]
  ip = IPython.get_ipython()
  # Text-and-replace seems stupid and you might think of doing:
  #   ip.user_ns[placeholder] = ip.user_ns[first_df]
  # But this doesn't work if the placeholder is assigned.
  cell_first_df = cell.replace(placeholder, first_df)
  ip.run_cell_magic('time_cell', "", cell_first_df)
  time1 = ip.user_ns['_TIMED_CELL']
  cell_second_df = cell.replace(placeholder, second_df)
  ip.run_cell_magic('time_cell', "", cell_second_df)
  time2 = ip.user_ns['_TIMED_CELL']
  ip.user_ns['_TIMED_CELL1'] = time1
  ip.user_ns['_TIMED_CELL2'] = time2
