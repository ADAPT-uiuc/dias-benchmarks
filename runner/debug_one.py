import run_nb

modin_cores = -1
less_replication = False
measure_mem = False

succ = run_nb.run_nb_paper(
  "../notebooks/lextoumbourou/feedback3-eda-hf-custom-trainer-sift",
    modin_cores, less_replication, measure_mem)
assert succ
