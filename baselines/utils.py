import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import loguniform, uniform, randint


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


def print_info(est, n_obs, n_var, hparams, file_name=None):
    print("------Result------")
    print("Time used: ", est)
    print("N observations: ", n_obs)
    print("N variables: ", n_var)
    print("Hyperparameters: ", hparams)
    if file_name is not None:
        new_df = pd.DataFrame({"new_result_0": {"time": est,
                                                "n_obs": n_obs,
                                                "n_var": n_var,
                                                "hparams": hparams}}).T
        if Path(file_name).is_file():
            df = pd.read_csv(file_name, index_col=0)
            df = pd.concat([df, new_df], axis=0)
            df.index = [f"new_result_{i}" for i in range(df.shape[0])]
        else:
            df = new_df
        df.to_csv(file_name)
