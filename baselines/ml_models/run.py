from lightgbm import LGBMRegressor
from sklearn.metrics import matthews_corrcoef, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import scanpy as sc

import argparse
from time import time
from pathlib import Path

from tqdm import tqdm
from baselines.utils import *


model_dic = {"lr": LinearRegression,
             "ridge": Ridge,
             #"xgb": XGBRegressor,
             "lgb": LGBMRegressor}


def train_and_predict(X_train,
                      y_train,
                      X_test,
                      y_test,
                      params,
                      model_name="ridge",
                      skiped_genes=None,
                      path_to_save=None
                      ):
    result_dic = {}
    skiped_genes = skiped_genes if skiped_genes is not None else []
    y_train_list, y_test_list, y_train_pred_list, y_test_pred_list = [], [], [], []
    for i, row in tqdm(y_train.var["gene_id"].items()):
        if row in skiped_genes:
            continue
        X, y = X_train, y_train[:, row]
        if X.shape[1] == 0:
            print(row, " not found in the transcriptomic data")
            continue
        X_test_part, y_test_part = X_test, y_test[:, row]
        if model_name == "lgb":
            model = model_dic[model_name](n_estimators=1000, random_state=42, **params)
        elif model_name == "ridge":
            model = model_dic[model_name](random_state=42, **params)
        else:
            model = model_dic[model_name](**params)
        model.fit(X.X.toarray(), np.ravel(y.X.toarray()))
        y_train_pred = np.expand_dims(model.predict(X.X.toarray()),axis=1)
        y_test_pred = np.expand_dims(model.predict(X_test_part.X.toarray()),axis=1)
        y_train_list.append(y.X.toarray())
        y_test_list.append(y_test_part.X.toarray())
        y_train_pred_list.append(y_train_pred)
        y_test_pred_list.append(y_test_pred)
        result_dic[row] = {
            "pearsonr_train_mean": pearsonr(y.X.toarray()[:, 0], y_train_pred[:, 0])[0],
            "pearsonr_test_mean": np.mean([pearsonr(y_test_part.X.toarray()[:, 0], y_test_pred[:, 0])[0]]),
            "rmse_train_mean": mean_squared_error(y.X.toarray(), y_train_pred),
            "rmse_test_mean": mean_squared_error(y_test_part.X.toarray(), y_test_pred)
        }
        if path_to_save is not None:
            pd.DataFrame(result_dic).to_csv(path_to_save)
    y_train_concat = np.concatenate(y_train_list, axis=1)
    y_train_pred_concat = np.concatenate(y_train_pred_list, axis=1)
    y_test_concat = np.concatenate(y_test_list, axis=1)
    y_test_pred_concat = np.concatenate(y_test_pred_list, axis=1)
    result_dic["All"] = {
      "pearsonr_train_mean": np.mean([pearsonr(y_train_concat[i, :], y_train_pred_concat[i, :])[0] for i in range(y_train.obs.shape[0])]),
      "pearsonr_test_mean": np.mean([pearsonr(y_test_concat[i, :], y_test_pred_concat[i, :])[0] for i in range(y_test.obs.shape[0])]),
      "pearsonr_train_std": np.std([pearsonr(y_train_concat[i, :], y_train_pred_concat[i, :])[0] for i in range(y_train.obs.shape[0])]),
      "pearsonr_test_std": np.std([pearsonr(y_test_concat[i, :], y_test_pred_concat[i, :])[0] for i in range(y_test.obs.shape[0])]),
      "rmse_train_mean": np.mean([mean_squared_error(y_train_concat[i, :], y_train_pred_concat[i, :]) for i in range(y_train.obs.shape[0])]),
      "rmse_test_mean": np.mean([mean_squared_error(y_test_concat[i, :], y_test_pred_concat[i, :]) for i in range(y_test.obs.shape[0])]),
      "rmse_train_std": np.std([mean_squared_error(y_train_concat[i, :], y_train_pred_concat[i, :]) for i in range(y_train.obs.shape[0])]),
      "rmse_test_std": np.std([mean_squared_error(y_test_concat[i, :], y_test_pred_concat[i, :]) for i in range(y_test.obs.shape[0])]),
    }
    return pd.DataFrame(result_dic), \
           pd.DataFrame(y_train_pred_concat,
                        columns=y_train.var.index,
                        index=y_train.obs.index), \
           pd.DataFrame(y_test_pred_concat,
                        columns=y_test.var.index,
                        index=y_test.obs.index)


def train_eval_time(X_train,
                    y_train,
                    params,
                    model_name="ridge"):
    start_time = time()
    for i, row in tqdm(y_train.var["gene_id"].items()):
        X, y = X_train, y_train[:, row]
        if model_name == "lgb":
            model = model_dic[model_name](n_estimators=1000, random_state=42, **params)
        elif model_name == "ridge":
            model = model_dic[model_name](random_state=42, **params)
        else:
            model = model_dic[model_name](**params)
        model.fit(X.X.toarray(), np.ravel(y.X.toarray()))
    return time() - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--io", help="Path to io configs", default='io')
    parser.add_argument("-o", "--obs", help="Number of observations", default='300')
    parser.add_argument("-v", "--var", help="Number of variables", default='1000')
    parser.add_argument("-p", "--hparam", help="Path to hparam configs", default='lgb_hparams')
    parser.add_argument("-e", "--eval_time", help="To evaluate training time or not",
                        action='store_true')
    parser.add_argument("-n", "--model", help="Name of trained model, choose from ['lgb', 'lr', 'ridge']",
                        default="lgb")
    parser.add_argument("-m", "--eval_memory", help="To evaluate memory usage or not",
                        action='store_true')

    args = parser.parse_args()

    cfd = Path(__file__).resolve().parent.parent / "configs"
    data_configs = load_config((cfd / args.io).with_suffix(".yml")) \
        if (cfd / args.io).with_suffix(".yml").is_file() else load_config(args.io)

    X_train = sc.read_h5ad(data_configs["X_train_pth"])
    y_train = sc.read_h5ad(data_configs["y_train_pth"])

    if args.eval_time:
        if args.model == "lgb":
            best_hparams = load_config((cfd / args.hparam).with_suffix(".yml")) \
                if (cfd / args.hparam).with_suffix(".yml").is_file() else load_config(args.hparam)
        else:
            best_hparams = {}
        X_train, y_train = get_subset(X_train,
                                      y_train,
                                      n_obs=int(args.obs),
                                      n_vars=int(args.var))
        if args.eval_memory:
            with open(Path(data_configs["result_dir_pth"]) / data_configs["mem_file_name"], "w+") as fp:
                est_time = train_eval_time_mem(train_eval_time, fp)(X_train=X_train,
                                                                    y_train=y_train,
                                                                    params=best_hparams,
                                                                    model_name=args.model)
            est_mem = extract_peak_mem(Path(data_configs["result_dir_pth"]) / data_configs["mem_file_name"])
        else:
            est_time = train_eval_time(X_train=X_train,
                                      y_train=y_train,
                                      params=best_hparams,
                                      model_name=args.model)
            est_mem = None
        result_dir = Path(data_configs["result_dir_pth"]).mkdir(parents=True, exist_ok=True)
        print_info(est_time, n_obs=int(args.obs), n_var=int(args.var), hparams=best_hparams, memory_used=est_mem,
                   file_name= Path(data_configs["result_dir_pth"]) / f"{args.model}.csv")
    else:
        if args.model == "lgb":
            used_hparams = load_config((cfd / args.hparam).with_suffix(".yml")) \
                if (cfd / args.hparam).with_suffix(".yml").is_file() else load_config(args.hparam)
        else:
            used_hparams = {}
        X_test = sc.read_h5ad(data_configs["X_test_pth"])
        y_test = sc.read_h5ad(data_configs["y_test_pth"])
        train_and_predict(X_train,
                          y_train,
                          X_test,
                          y_test,
                          params=used_hparams,
                          model_name=args.model)