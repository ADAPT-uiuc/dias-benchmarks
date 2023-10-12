import run_nb
import argparse

provided_args = {
    'alt': None,
    'cores': 12,
    'less_replication': True,
    'measure_mem': False
}

# Parse the dictionary of argument values
args = argparse.Namespace(**provided_args)

succ = run_nb.run_nb_paper(
  "../notebooks/joshuaswords/netflix-data-visualization/", args)
assert succ
