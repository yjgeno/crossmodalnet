from baselines.sciPENN.src.sciPENN_API import sciPENN_API

import scanpy as sc
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, mean_squared_error, r2_score
from scipy.stats import pearsonr

from pathlib import Path
from time import time
import argparse
from baselines.utils import *


categorical_ftrs_1 = ["donor", "day"]
categorical_ftrs_2 = ["sample", "timepoint"]


def train_eval_time(X_train,
                    y_train,
                    params):
    if all([c in X_train.obs.columns for c in categorical_ftrs_1]):
        train_batch_keys = categorical_ftrs_1
        test_batchkey = categorical_ftrs_1[0]
    elif all([c in X_train.obs.columns for c in categorical_ftrs_2]):
        train_batch_keys = categorical_ftrs_2
        test_batchkey = categorical_ftrs_2[0]
    else:
        raise ValueError("X_train must contains either [donor, day] or [sample, timepoint]")

    for k in train_batch_keys:
        X_train.obs[k] = X_train.obs[k].astype(str).astype('category')
        y_train.obs[k] = y_train.obs[k].astype(str).astype('category')
        #X_test.obs[k] = X_test.obs[k].astype(str)

    sciPENN = sciPENN_API(gene_trainsets=[X_train],
                          protein_trainsets=[y_train],
                          train_batchkeys=train_batch_keys,
                          cell_normalize=False,
                          log_normalize=False,
                          gene_normalize=True,
                          min_cells=0,
                          min_genes=0,
                          type_key='cell_type')
    s_time = time()
    sciPENN.train(load=False, **params)
    return time() - s_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--io", help="Path to io configs", default='io')
    parser.add_argument("-o", "--obs", help="Number of observations", default='300')
    parser.add_argument("-v", "--var", help="Number of variables", default='1000')
    parser.add_argument("-p", "--hparam", help="Path to hparam configs", default='sciPENN_hparams')
    parser.add_argument("-e", "--eval_time", help="To evaluate training time or not",
                        action='store_true')
    parser.add_argument("-m", "--eval_memory", help="To evaluate memory usage or not",
                        action='store_true')
    args = parser.parse_args()

    cfd = Path(__file__).resolve().parent.parent / "configs"
    data_configs = load_config((cfd / args.io).with_suffix(".yml")) \
        if (cfd / args.io).with_suffix(".yml").is_file() else load_config(args.io)

    X_train = sc.read_h5ad(data_configs["X_train_pth"])
    y_train = sc.read_h5ad(data_configs["y_train_pth"])

    if args.eval_time:
        best_hparams = load_config((cfd / args.hparam).with_suffix(".yml")) \
            if (cfd / args.hparam).with_suffix(".yml").is_file() else load_config(args.hparam)
        X_train, y_train = get_subset(X_train,
                                      y_train,
                                      n_obs=int(args.obs),
                                      n_vars=int(args.var))
        X_train.var.index = X_train.var.index + pd.Series([i for i in range(X_train.var.shape[0])]).astype(str)
        y_train.var.index = y_train.var.index + pd.Series([i for i in range(y_train.var.shape[0])]).astype(str)

        X_train.obs.index = X_train.obs.index + pd.Series([i for i in range(X_train.obs.shape[0])]).astype(str)
        y_train.obs.index = y_train.obs.index + pd.Series([i for i in range(y_train.obs.shape[0])]).astype(str)

        result_dir = Path(data_configs["result_dir_pth"]).mkdir(parents=True, exist_ok=True)
        if args.eval_memory:
            with open(Path(data_configs["result_dir_pth"]) / data_configs["mem_file_name"], "w+") as fp:
                est_time = train_eval_time_mem(train_eval_time, fp)(X_train=X_train,
                                                                    y_train=y_train,
                                                                    params=best_hparams)
            est_mem = extract_peak_mem(Path(data_configs["result_dir_pth"]) / data_configs["mem_file_name"])
            
        else:
            est_time = train_eval_time(X_train=X_train,
                                    y_train=y_train,
                                    params=best_hparams)
            est_mem = None
        print_info(est_time,
                   n_obs=int(args.obs),
                   n_var=int(args.var),
                   hparams=best_hparams,
                   memory_used=est_mem,
                   file_name= Path(data_configs["result_dir_pth"]) / "scPENN.csv")
    else:
        used_hparams = load_config((cfd / args.hparam).with_suffix(".yml")) \
            if (cfd / args.hparam).with_suffix(".yml").is_file() else load_config(args.hparam)
        X_test = sc.read_h5ad(data_configs["X_test_pth"])
        y_test = sc.read_h5ad(data_configs["y_test_pth"])
        train_and_predict(X_train,
                          y_train,
                          X_test,
                          y_test,
                          params=used_hparams,
                          model_name=args.model)
