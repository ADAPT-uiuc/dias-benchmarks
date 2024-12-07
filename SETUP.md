## Ideal System

The system we ran the experiments on had:
- 12-core AMD Ryzen 5900X
- 32GB of main memory
- Samsung 980 PRO NVMe SSD, 256GB
- Ubuntu 22.04.1 LTS.

To reproduce the results as accurately as possible, please do not run on a VM
and use a fast disk. That said, we have reproduced the results on AWS instances,
and the SIGMOD ARI folks seem to have reproduced them using a VM.

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

Set the environmental variable `DIAS_ROOT` to where the folder `dias` is.
```bash
export DIAS_ROOT=<dias root>
```

For example, if you unzipped `artifact.zip` to `/home/foo/artifact`, then `DIAS_ROOT`
should be set to:
```bash
export DIAS_ROOT=/home/foo/artifact/dias
```

### Install library dependencies

Note that in the following, as above, if you unzipped `artifact.zip` to
`/home/foo/artifact`, then `<dias-benchmarks root>` is `/home/foo/artifact/dias-benchmarks`:

```bash
cd <dias-benchmarks root>/runner
pip install -r requirements.txt
```

### Download the datasets

Instead of downloading the datasets zip, you can use the datasets zip we provide
with the ARI submission.

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

## Pre-Run

```bash
cd <dias-benchmarks root>/runner
```

### Quiescing the machine

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