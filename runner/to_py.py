# Converst .ipynb to .py with some wrappers around the cells to e.g., call the
# rewriter, measure etc. This is similar to jupyter nbconvert --to script but
# much more specific.
#
# This doesn't work for any notebook, only for the notebooks used in the
# evaluation. We can make several assumptions about those, like e.g., that they
# don't include magic functions.
#
# The .py should still run with IPython. The reason is that we create
# dynamically generated code (the result of rewrite_ast()), which we need to
# execute. In Python, the standard way is to use exec(). The problem is if
# functions are defined in this dynamically generated, we cannot retrieve its
# source code using inspect.getsource(), which we potentially need to do in
# _DIAS_apply() (in the rewriter). Using IPython, we can execute code using
# get_ipython().run_cell(), which for some reason doesn't have the same problem.
# I don't think it makes sense to spend time to investigate how to do it in pure
# Python. The only reason would be to save something in overheads, and I don't
# think the savings will be big.

#
# TODO: Currently, it's unclear whether this is the best way to measure. Maybe
# it's better to the rewrite "offline" to generate the rewritten code, and run
# that as a Python file. This cannot be done for the apply() patterns though,
# which is why I feel this is better because it measures every pattern with the
# same yardstick.
#
# TODO: Intersperse calls to `ray memory`. The main problem of the
# ray_memory_sampler.sh is that it runs in parallel. It calls `ray memory` which
# is very slow to query. Because it is running in parallel, the notebook doesn't
# wait for queries to finish and we may miss a query that would give us a bigger
# memory consumption, thereby missing the actual max memory consumption. If we
# intersperse _inside_ the .py, that won't be a problem. However, this may
# create another problem which is that this may tamper with the runtime
# measurements of the actual operations in the notebook. For example, it's
# possible that calling `ray memory` will clear the cache, which would otherwise
# allow for locality later in the notebook. So, we should do a different run for
# memory and runtime measurements, where, for the runtime measurements, we won't
# include the calls to `ray memory`.

#----------------------------------------
import bench_utils
import sys

nb_path = sys.argv[1]
assert nb_path.endswith(".ipynb")
base_path = nb_path.split('.ipynb')[0]
py_file = open(f"{base_path}.py", "w")
base_name = base_path.split('/')[-1]

def gen(s):
  py_file.write(s + "\n")

def cell_doesnt_contain_magic(cell):
  for line in cell.splitlines():
    if line.strip().startswith("%"):
      return False
  # END FOR #
  return True

# imports
gen("import dias.rewriter")
gen("import time")
gen("import sys")
gen("import json")
gen("from IPython import get_ipython")

gen("""
_DIAS_ip = get_ipython()
if _DIAS_ip is None:
  print("IPython is required")
  sys.exit(1)
""")

# Introduce cell stats
gen("_DIAS_cell_stats = []")

source_cells = bench_utils.open_and_get_source_cells(nb_path)
# Just convenience to match jupyter nbconvert. Some editors recognize
# it.
cell_begin = "# In[ ]:"

for cell in source_cells:
  assert cell_doesnt_contain_magic(cell)
  # Escape it because it may contain triple quotes.
  cell = cell.replace('"""', '\\"\\"\\"')

  # Make the code call rewrite_ast()
  new_cell = f"""
_DIAS_raw_source = \
\"\"\"{cell}\"\"\"
dias.rewriter._DIAS_apply_overhead_ns = 0
dias.rewriter._DIAS_apply_pat = None
## Patt match and rewrite
_DIAS_rewrite_start = time.perf_counter_ns()
_DIAS_new_source, _DIAS_patts_hit = dias.rewriter.rewrite_ast_from_source(_DIAS_raw_source)
_DIAS_rewrite_end = time.perf_counter_ns()
_DIAS_rewrite_ns = _DIAS_rewrite_end - _DIAS_rewrite_start

## Execute
_DIAS_exec_start = time.perf_counter_ns()
_DIAS_ip.run_cell(_DIAS_new_source)
_DIAS_exec_end = time.perf_counter_ns()
_DIAS_exec_ns = _DIAS_exec_end - _DIAS_exec_start

# Finish stats
_DIAS_overhead_ns = _DIAS_rewrite_ns + dias.rewriter._DIAS_apply_overhead_ns

_DIAS_stats = dict()
_DIAS_stats['raw'] = _DIAS_raw_source
_DIAS_stats['rewrite-ns'] = _DIAS_rewrite_ns
_DIAS_stats['overhead-ns'] = _DIAS_overhead_ns
_DIAS_stats['exec-ns'] = _DIAS_exec_ns
_DIAS_stats['total-ns'] = _DIAS_rewrite_ns + _DIAS_exec_ns
if dias.rewriter._DIAS_apply_pat != None:
  _DIAS_the_pat = dias.rewriter._DIAS_apply_pat
  _DIAS_patts_hit[_DIAS_the_pat.name] = 1
_DIAS_stats['patts-hit'] = _DIAS_patts_hit

_DIAS_cell_stats.append(_DIAS_stats)
"""
  
  gen(new_cell)

# END FOR LOOP #

## Testing
gen("print(_DIAS_cell_stats)")

## Output the stats to a JSON file
gen(f"""
out_json = dict()
out_json['cells'] = _DIAS_cell_stats
json_f = open("{base_name}.json", "w")
json.dump(out_json, fp=json_f, indent=2)
json_f.close()
""")

py_file.close()