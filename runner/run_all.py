import subprocess
from glob import glob
import run_nb
import argparse
import os
import sys
import pathlib

parser = argparse.ArgumentParser(description='Run all benchmarks')
parser.add_argument('--rewriter', action='store_true', help='Run with the rewriter enabled.')
# For the default number of cores in Modin: https://github.com/modin-project/modin/blob/b998925d9e34bdb5c0752abb85a7a5769e0826f1/modin/config/envvars.py#L215)
parser.add_argument('--modin', default=-1, type=int, metavar='NUM_CORES', help='If not specified, we use vanilla Pandas. Otherwise, we use Modin Pandas using NUM_CORES cores (Modin\'s default is your machine\'s number of threads).')
parser.add_argument('--less_replication', action='store_true', help='Less replication of data.')
parser.add_argument('--measure_modin_mem', action='store_true', help='Only measure memory consumption of Modin.')

args = parser.parse_args()

# Some validation
msg=None
if args.modin != -1 and args.modin < 2:
  msg = "NUM_CORES for option --modin must be at least 2"

if msg is not None:
  print("ERROR:", msg)
  parser.print_help()
  sys.exit(1)

# Put a version file into the "stats" folder
assert os.path.isdir("./stats")
ver_file = open('stats/.version', 'w+')
VER_rewriter = "rewr_ON" if args.rewriter else "rewr_OFF"
VER_modin = "modin_OFF" if args.modin == -1 else f"modin_{args.modin}"
VER_repl = "repl_LESS" if args.less_replication else "repl_STD"
VER_sliced_exec = "modin_mem_ON" if args.measure_modin_mem else "modin_mem_OFF"
VER="-".join((VER_rewriter, VER_modin, VER_repl, VER_sliced_exec))
ver_file.write(VER)
ver_file.close()

prefix = str(pathlib.Path('../notebooks').resolve())

nbs_we_hit = [
  "lextoumbourou/feedback3-eda-hf-custom-trainer-sift",
  "paultimothymooney/kaggle-survey-2022-all-results",
  "dataranch/supermarket-sales-prediction-xgboost-fastai",
  "kkhandekar/environmental-vs-ai-startups-india-eda",
  "ampiiere/animal-crossing-villager-popularity-analysis",
  "aieducation/what-course-are-you-going-to-take",
  "saisandeepjallepalli/adidas-retail-eda-data-visualization",
  "joshuaswords/netflix-data-visualization",
  "spscientist/student-performance-in-exams",
 "ibtesama/getting-started-with-a-movie-recommendation-system",
]

nbs_we_dont = [
  "nickwan/creating-player-stats-using-tracking-data",
  "erikbruin/nlp-on-student-writing-eda",
  "madhurpant/beautiful-kaggle-2022-analysis",
  "pmarcelino/comprehensive-data-exploration-with-python",
  "gksriharsha/eda-speedtests",
  "mpwolke/just-you-wait-rishi-sunak",
  "sanket7994/imdb-dataset-eda-project",
  "roopacalistus/retail-supermarket-store-analysis",
  "sandhyakrishnan02/indian-startup-growth-analysis",
  "roopacalistus/exploratory-data-analysis-retail-supermarket"
]

nbs = nbs_we_hit + nbs_we_dont

for nb in nbs:
  kernel_user = nb.split('/')[0]
  kernel_slug = nb.split('/')[1]
  full_path = prefix+"/"+nb
  print(f"--- RUNNING: {kernel_user}/{kernel_slug}")
  succ = run_nb.run_nb_paper(full_path, args.rewriter, args.modin, args.less_replication, args.measure_modin_mem)
  assert succ
  res = subprocess.run(["mv", "stats.json", f"stats/{kernel_user}_{kernel_slug}.json"])
  assert res.returncode == 0
