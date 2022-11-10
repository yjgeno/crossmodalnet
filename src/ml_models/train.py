import argparse
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import joblib

from regressions import *
from utils import *

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import warnings


def preprocess(train_X,
               test_X,
               train_X_path,
               test_X_path,
               pipeline_path) -> None:
    fitted = False
    if Path(pipeline_path).is_file():
        print("loading fitted preprocessor")
        pl = joblib.load(pipeline_path)
        fitted = True
    else:
        pl = Pipeline([("std", StandardScaler()),
                       ("svd", TruncatedSVD(n_components=512, n_iter=20, random_state=1))])
    if not Path(train_X_path).is_file() or not fitted:
        if fitted:
            train_X = pl.transform(train_X)
        else:
            print("fitting preprocessor")
            train_X = pl.fit_transform(train_X)
            joblib.dump(pl, pipeline_path)
        with open(train_X_path, 'wb') as f:
            np.save(f, train_X)
    if not Path(test_X_path).is_file():
        test_X = pl.transform(test_X)
        with open(test_X_path, 'wb') as f:
            np.save(f, test_X)


def check_processed(train_X_raw,
                    train_X_pps,
                    test_X_raw,
                    test_X_pps,
                    pipeline_path,
                    ):
    if Path(train_X_pps).is_file() and Path(test_X_pps).is_file():
        return
    train_X = sc.read_h5ad(train_X_raw).X.toarray() if not Path(train_X_pps).is_file() else None
    test_X = sc.read_h5ad(test_X_raw).X.toarray() if not Path(test_X_pps).is_file() else None
    preprocess(train_X,
               test_X,
               train_X_path=train_X_pps,
               test_X_path=test_X_pps,
               pipeline_path=pipeline_path)


def train(io_config,
          model_config,
          cv_config,
          n_split,
          use_gpu):

    cv_clfs = []
    assert Path(io_config["input_training_x"]).is_file()
    assert Path(io_config["input_training_y"]).is_file()
    assert Path(io_config["input_test_x"]).is_file()

    check_processed(train_X_raw=Path(io_config["input_training_x"]),
                    train_X_pps=Path(io_config["input_pps_training_x"]),
                    test_X_raw=Path(io_config["input_test_x"]),
                    test_X_pps=Path(io_config["input_pps_test_x"]),
                    pipeline_path=Path(io_config["input_preprocessor_x"]),
                    )
    ftr_indexes = []

    for s in range(n_split):
        # data loading
        with open(io_config["input_pps_training_x"], 'rb') as f:
            train_X = np.load(f)
        n_cells, n_ftrs = train_X.shape
        sample_idx = np.random.choice(n_cells, n_cells // n_split)  if n_split != 1 else np.arange(0, n_cells)
        ftr_indexes.append(sample_idx)
        train_X = train_X[sample_idx, :]
        train_y = sc.read_h5ad(io_config["input_training_y"]).X[sample_idx, :]
        y_var_len = train_y.shape[1]

        output_dir = Path(io_config["output_dir"])
        output_dir = Path(output_dir / f"{model_config['model_name']}")
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / f"ftr_ind_{s}.npy", 'wb') as f:
            np.save(f, sample_idx)

        # init
        cv_object = Regressor(reg_name=model_config["model_name"],
                              use_cuml=use_gpu)

        # training
        for y_ind in range(y_var_len):
            cv_object.partial_cv(train_X, train_y[:, y_ind].toarray().ravel(),
                                 y_ind, param_dist=model_config["param"], **cv_config)
        cv_object.save_iters(output_dir / f"cv_result_{s}.csv")

        if use_gpu:
            (output_dir / f"models").mkdir(parents=True, exist_ok=True)
            cv_object.save_model(output_dir / f"models")
        else:
            cv_object.save_model(output_dir / f"best_model_{s}.joblib")
        cv_clfs.append(cv_object)
        del train_X
        del train_y

    # prediction
    with open(io_config["input_pps_test_x"], 'rb') as f:
        test_X = np.load(f)

    for i, clf in enumerate(cv_clfs):
        predicted_y = clf.predict(test_X[ftr_indexes[i], :])
        pd.DataFrame(data=predicted_y).to_csv(output_dir / f"predicted_y_{i}.csv")

    del test_X
    pred_y_mean = np.zeros(predicted_y.shape)
    for i in range(len(cv_clfs)):
        pred_y_mean += pd.read_csv(output_dir / f"predicted_y_{i}.csv", index_col=0).values
    pred_y_mean /= len(cv_clfs)
    pd.DataFrame(data=pred_y_mean).to_csv(output_dir / f"predicted_y_mean.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cfd = Path(__file__).resolve().parent / "configs"

    parser.add_argument("-m", "--model", help="model config file name or location")
    parser.add_argument("-p", "--problem", default="cite")
    parser.add_argument("-i", "--io", help="IO config file name or location", default="test_io_cfg")
    parser.add_argument("-c", "--cv", help="cv config file name or location", default="cv_cfg")
    parser.add_argument("-n", "--num", help="Number of bootstrap samplings", default=5, type=int)
    parser.add_argument("-g", "--gpu", help="Use GPU", action='store_true')

    args = parser.parse_args()
    assert args.problem in ["cite", "multi"]
    model_cfd = cfd / f"models_for_{args.problem}"

    warnings.filterwarnings(action='ignore', category=FutureWarning)

    print("Config dir located: ", model_cfd.is_dir())
    cfg_name = args.model + ("_g" if args.gpu else "")

    model_configs = load_config((model_cfd / cfg_name).with_suffix(".yml")) \
        if (model_cfd / cfg_name).with_suffix(".yml").is_file() else load_config(cfg_name)
    io_configs = load_config((cfd / args.io).with_suffix(".yml")) \
        if (cfd / args.io).with_suffix(".yml").is_file() else load_config(args.io)
    cv_configs = load_config((cfd / args.cv).with_suffix(".yml")) \
        if (cfd / args.cv).with_suffix(".yml").is_file() else load_config(args.cv)

    model_name = model_configs.pop("model")
    #fs_name = model_configs.pop("fs_name")
    #fs_est = model_configs.pop("fs_estimator")
    train(io_config=io_configs,
          model_config={"model_name": model_name,
                        "param": parse_config(model_configs, use_gpu=args.gpu)},
          cv_config=cv_configs,
          n_split=args.num,
          use_gpu=args.gpu)
