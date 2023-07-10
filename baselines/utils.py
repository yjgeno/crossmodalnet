import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import re
from scipy.stats import loguniform, uniform, randint
from memory_profiler import profile
import scanpy as sc
from anndata import AnnData
from scipy.io import mmwrite, mmread
from pathlib import Path


class H5adToMtx:
    def __init__(self, X_path) -> None:
        self.data = sc.read_h5ad(X_path)

    def _process_var(self, var, func):
        if func is not None:
            func(var)
        return var
    
    def _process_obs(self, obs, func):
        if func is not None:
            func(obs) 
        return obs
    
    def run(self, save_path, transport=True, var_func=None, obs_func=None):
        if transport:
            mtx = self.data.X.T
        else:
            mtx = self.data.X
        
        mmwrite(str(Path(save_path) / "x.mtx"), mtx)
        var = self._process_var(self.data.var, var_func)
        obs = self._process_obs(self.data.obs, obs_func)
        var.to_csv(Path(save_path) / "var.csv")
        obs.to_csv(Path(save_path) / "obs.csv")


def load_mtx_dir(mtx, var, obs):
    with open(mtx) as f:
        x = mmread(f)
    return AnnData(X=x, var=pd.read_csv(var, index_col=0), obs=pd.read_csv(obs, index_col=0))


def get_subset(X_data, y_data, n_obs, n_vars, random_state=42):
    rng = np.random.default_rng(random_state)
    obs_idx = rng.choice(X_data.obs.shape[0], n_obs)
    var_idx = rng.choice(X_data.var.shape[0], n_vars)
    return X_data[obs_idx, var_idx], y_data[obs_idx, :]


def load_config(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def parse_config(config, use_gpu) -> dict:
    dist = {"i": randint, "l": loguniform, "f": uniform}
    params = {}
    print(config)
    for k, v in config.items():
        if isinstance(v, dict) and v["type"] == "c":
            params[f"{k}" if use_gpu else f"{k}"] = v["choices"]
        else:
            params[f"{k}" if use_gpu else f"{k}"] = dist[v["type"]](v["lo"], v["hi"])
    return params


def print_info(est, n_obs, n_var, hparams, memory_used=None, file_name=None):
    print("------Result------")
    print("Time used: ", est)
    print("N observations: ", n_obs)
    print("N variables: ", n_var)
    print("Hyperparameters: ", hparams)
    if memory_used is not None:
        print("Max memory used: ", memory_used, "MiB")

    if file_name is not None:

        new_df = pd.DataFrame({"new_result_0": {"time": est,
                                                "n_obs": n_obs,
                                                "n_var": n_var,
                                                "hparams": hparams,
                                                "memory_used (MiB)": memory_used}}).T
        if Path(file_name).is_file():
            df = pd.read_csv(file_name, index_col=0)
            df = pd.concat([df, new_df], axis=0)
            df.index = [f"new_result_{i}" for i in range(df.shape[0])]
        else:
            df = new_df
        df.to_csv(file_name)


def train_eval_time_mem(train_func, memprof_file):
    return profile(func=train_func, stream=memprof_file)


def extract_peak_mem(data_file_name):
    with open(data_file_name) as fp:
        data = fp.readlines()
    mems = []
    for line in data:
        pattern = r"[0-9]+\.\d\sMiB"
        matched = re.findall(pattern, line)
        if len(matched):
            mems.append(float(matched[0][:matched[0].index(" MiB")]))
    return max(mems)