from pathlib import Path
import nbformat

def main():
    notebooks_path = Path('notebooks')
    for nb_path in notebooks_path.glob('**/*.ipynb'):
        print(nb_path)
        with open(nb_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        nb['cells'].insert(2, nbformat.v4.new_code_cell(source="%load_ext autotime"))
        # snappy_path_cell = nbformat.v4.new_code_cell(source='import os\nSNAPPY_notebook_path = os.path.join(os.path.abspath(""), "bench.ipynb")')
        # time_start_cell = nbformat.v4.new_code_cell(source='import time\nSNAPPY_start_time = time.perf_counter_ns()')
        # time_end_cell = nbformat.v4.new_code_cell(source='SNAPPY_end_time = time.perf_counter_ns()\nprint("Total elapsed time:", (SNAPPY_end_time - SNAPPY_start_time) / (10 ** 9))')
        # nb['cells'].insert(0, time_start_cell)
        # nb['cells'].insert(0, snappy_path_cell)
        # nb['cells'].insert(-1, time_end_cell)
        with open(nb_path, 'w') as f:
            nbformat.write(nb, f)


if __name__ == '__main__':
    main()