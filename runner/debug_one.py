import run_nb

enable_rewriter = False
with_modin = True
less_replication = True
measure_modin_mem = True

succ = run_nb.run_nb(
  "<notebooks root>/<user>/<nb>",
  enable_rewriter, with_modin, less_replication, measure_modin_mem)
assert succ
