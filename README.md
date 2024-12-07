## Main Experiments (Sections 6.2 and 6.3)

In the following, `<dias-benchmarks root>` is the root of `dias-benchmarks`. For example, if you extract `10.1145_3639313_source_code.zip` in the directory `/home/foo`, you should get the directory `/home/foo/artifact`. Then, `<dias-benchmarks root>` is: `/home/foo/artifact/dias-benchmarks`.

To produce all the results needed you only need to run two scripts. First, `cd <dias-benchmarks root>` and run the following. **It's necessary to `source` the script!**:
```bash
source setup_everything.sh
```

This should set up everything needed to run the experiments, but in case it fails, you can consult `<dias-benchmarks root>/SETUP.md`.

To run the experiments in the paper, `cd <dias-benchmarks root>/runner` and run:

```bash
./paper_exps.sh
```

### Producing the plots

Running `paper_exps.sh` should produce the following directories:
```
stats-rewr_OFF-modin_12-repl_LESS-sliced_exec_ON
stats-rewr_OFF-modin_4-repl_LESS-sliced_exec_ON
stats-rewr_OFF-modin_8-repl_LESS-sliced_exec_ON
stats-rewr_OFF-modin_OFF-repl_LESS-sliced_exec_ON
stats-rewr_OFF-modin_OFF-repl_STD-sliced_exec_ON
stats-rewr_ON-modin_OFF-repl_LESS-sliced_exec_ON
stats-rewr_ON-modin_OFF-repl_STD-sliced_exec_OFF
stats-rewr_ON-modin_OFF-repl_STD-sliced_exec_ON
stats-rewr_stats
```

We move those to `<dias-benchmarks root>/stats` (where `stats.ipynb` is) e.g., with:
```bash
cd <dias-benchmarks root>
mv runner/stats* stats
```

Then, you can just run `stats.ipynb` to produce the plots.

## Comparing Various Dataframe Libraries (Section 6.6)

The directory `<dias-benchmarks root>/pandas-replacements` is independent and ha the code used to
produce the numbers in Table 2 of the paper. Producing the numbers and the table
was done manually (FYI, the table was created with Google Docs).
