from pathlib import Path
import json
import statistics
import random
random.seed(8980)
def main():
    notebooks_path = Path(__file__).parent / 'notebooks'
    good_notebooks = []
    numeric_speedups = []
    relative_speedups = []
    for jp in notebooks_path.glob('**/out.json'):
        try:
            with open(jp, 'r') as f:
                blob = json.load(f)
            if len(blob) == 0:
                continue
        except json.JSONDecodeError:
            continue
        good_notebooks.append(str(jp))
        bench_py = blob['results'][0]
        bench_ipynb = blob['results'][1]
        print(jp.parent)
        print(bench_py['mean'], bench_ipynb['mean'])
        numeric_speedups.append(bench_ipynb['mean'] - bench_py['mean'])
        relative_speedups.append( bench_ipynb['mean'] / bench_py['mean'])
    print("Got", len(relative_speedups))
    print("RELATIVE")
    print("\tRange:", (min(relative_speedups), max(relative_speedups)))
    print("\tMean:", statistics.mean(relative_speedups))
    print("ABSOLUTE")
    print("\tRange:", (min(numeric_speedups), max(numeric_speedups)))
    print("\tMean:", statistics.mean(numeric_speedups))
    print('\n\n\t' + '\n\t'.join(random.sample(good_notebooks, 5)))
if __name__ == '__main__':
    main()