# This file is loaded as is in other files.

# Settings
NUM_CPUS = 12
ENGINE = "ray"

# Configure Modin and Ray
from modin.config import Engine, CpuCount
Engine.put(ENGINE)
CpuCount.put(NUM_CPUS)

assert ENGINE == "ray"
import ray
# NOTE: We suppress ray warnings
ray.init(num_cpus=NUM_CPUS, runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}}, log_to_driver=False)

# Import modules
import modin.pandas as modin_pd
import pandas as pd
import numpy as np