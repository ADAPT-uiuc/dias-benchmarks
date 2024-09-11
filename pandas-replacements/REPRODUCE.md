## AWS configuration

- Ubuntu 22.04
- c5.24xlarge (96 vCPUs, 192 GiB of RAM)
- 400GB SSD

## Reproducing the environment

### Install Python

```bash
# We use version 3.10.6
sudo apt install python3.10
```

### Install `pip`, `venv`
```bash
sudo apt install python3-pip
sudo apt install python3.10-venv
```

### Create environment (named `env`) and activate it
```bash
python3 -m venv env
source env/bin/activate
```

### Install dependencies
```bash
pip install -r <dias-benchmarks root>/pandas-replacements/requirements.txt
```

### Download datasets
```bash
<dias-benchmarks root>/pandas-replacements/datasets/download_datasets.sh
```

### Reproducing numbers
You should be able to run any of the notebooks with Jupyter. You should run a notebook multiple times
because Jupyter virtualizes many things, including e.g., the disk. Note that the numbers may be quite
different from run to run and from the paper. We could not avoid this variability. However, the _conclusions_
should be the same, namely the orders of magnitude of slowdowns and speedups mentioned in the paper should be the same.

### Visualization

Unfortunately, the visualization (Table 2 in the paper) was not produced with any script and I don't know how to create
one because I used Google Slides to create it. The number were copied manually from the notebook cells.