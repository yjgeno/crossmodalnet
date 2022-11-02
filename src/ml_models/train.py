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
          n_split):

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

    for s in range(n_split):
        # data loading
        with open(io_config["input_pps_training_x"], 'rb') as f:
            train_X = np.load(f)
        n_cells, n_ftrs = train_X.shape
        sample_idx = np.random.choice(n_cells, n_cells // n_split)

        train_X = train_X[sample_idx, :]
        train_y = sc.read_h5ad(io_config["input_training_y"]).X[sample_idx, :].toarray()

        output_dir = Path(io_config["output_dir"])
        output_dir = Path(output_dir / f"{model_config['model_name']}_{model_config['fs_name']}_{model_config['fs_est']}")
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        # init
        cv_object = Regressor(reg_name=model_config["model_name"],
                              fs_name=model_config["fs_name"],
                              fs_est=model_config["fs_est"])

        # training
        cv_object.cross_validation(train_X, train_y, param_dist=model_config["param"], **cv_config)
        cv_object.save_iters(output_dir / f"cv_result_{s}.csv")
        cv_object.save_model(output_dir / f"best_model_{s}.joblib")
        cv_clfs.append(cv_object)
        del train_X
        del train_y

    # prediction
    with open(io_config["input_pps_test_x"], 'rb') as f:
        test_X = np.load(f)

    for i, clf in enumerate(cv_clfs):
        predicted_y = clf.predict(test_X)
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

    args = parser.parse_args()
    assert args.problem in ["cite", "multi"]
    model_cfd = cfd / f"models_for_{args.problem}"

    print("Config dir located: ", model_cfd.is_dir())

    model_configs = load_config((model_cfd / args.model).with_suffix(".yml")) \
        if (model_cfd / args.model).with_suffix(".yml").is_file() else load_config(args.model)
    io_configs = load_config((cfd / args.io).with_suffix(".yml")) \
        if (cfd / args.io).with_suffix(".yml").is_file() else load_config(args.io)
    cv_configs = load_config((cfd / args.cv).with_suffix(".yml")) \
        if (cfd / args.cv).with_suffix(".yml").is_file() else load_config(args.cv)

    model_name = model_configs.pop("model")
    fs_name = model_configs.pop("fs_name")
    fs_est = model_configs.pop("fs_estimator")
    train(io_config=io_configs,
          model_config={"model_name": model_name,
                        "fs_name": fs_name,
                        "fs_est": fs_est,
                        "param": parse_config(model_configs)},
          cv_config=cv_configs,
          n_split=args.num)
