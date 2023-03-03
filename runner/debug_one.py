import run_nb

enable_rewriter = False
with_modin = True
less_replication = True

succ = run_nb.run_nb(
  "<notebooks root>/<user>/<nb>",
  enable_rewriter, with_modin, less_replication)
assert succ
