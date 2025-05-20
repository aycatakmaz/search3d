import os
import warnings
from tempfile import NamedTemporaryFile
import numpy as np
import torch
import glob
from datetime import date

def get_free_gpu(min_mem=20000):
    try:
        with NamedTemporaryFile() as f:
            os.system(f"nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > {f.name}")
            memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
        if max(memory_available) < min_mem:
            warnings.warn("Not enough memory on GPU, using CPU")
            return torch.device("cpu")
        return torch.device("cuda", np.argmax(memory_available))
    except:
        warnings.warn("Could not get free GPU, using CPU")
        return torch.device("cpu")
    
def create_out_folder(output_path: str = "outputs"):
    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join(output_path, date_str)
    experiment_paths = sorted(glob.glob(os.path.join(out_folder, "experiment_*")))
    n_experiment = len(experiment_paths)
    out_folder = os.path.join(out_folder, "experiment_" + str(n_experiment))
    os.makedirs(out_folder, exist_ok=True)
    return out_folder