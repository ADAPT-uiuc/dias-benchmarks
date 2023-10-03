## Setting Up the Python Environment

```bash
sudo apt-get update
```

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

### Create environment (named `env`) and Activate it
```bash
python3 -m venv env
source env/bin/activate
```

## Setup Artifact

```
### Download the benchmark infrastructure and code (this repo)
```bash
git clone https://github.com/baziotis/cs598ms-nbs --single-branch
```

### Install library dependencies
```bash
cd <cs598ms-nbs root>/runner
pip install -r requirements.txt
```

### Download the datasets

From Box:
```bash
wget https://uofi.box.com/shared/static/9r1fgjdpoz113ed2al7k1biwxgnn9fpa -O dias_datasets.zip
# This should create a directory named dias-datasets
unzip dias_datasets.zip
```

Alternatively, you can use the following Google Drive link:
```bash
https://drive.google.com/file/d/1IJjGO5OHllVcg0l8-wxYojmw4vFzvlY0/view?usp=share_link
```

### Copy the datasets to where the notebooks are
You can use the script `copier.sh` that comes with the dataset folder (i.e., `dias-datasets`). You need to run it from the folder it is in. You pass it one argument, where the notebooks root directory is. For example:
```bash
./copier.sh ~/dias-benchmarks/notebooks
```

## Run the Artifact

```bash
cd <dias-benchmarks root>/runner
```

### Quiescing the machine (Optional)

You can use the following script to quiesce the machine in the same way we did. If you have an Intel, open this script and modify it slightly. It has instructions.

Note that quiescing probably won't work if you're using a VM.
```bash
./quiesce.sh
```

### Pre-run script

**Make sure you have activated the `pip` environment created above**

```bash
./pre_run.sh
```

This requires `sudo` because we want to support writing in `/kaggle/working`, which is allowed on Kaggle and the notebooks we include use it. So, we create this directory.

### Running the experiments

You can review the `run_all.py` options with `python run_all.py --help` to see what different configurations can be run.

Here is an example:
```
python run_all.py --less_replication
```