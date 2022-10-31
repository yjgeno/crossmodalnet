import argparse
from pathlib import Path
import scanpy as sc
import pandas as pd

from .regressions import *
from .utils import *


def train(io_config,
          model_config,
          cv_config,
          n_split):

    cv_clfs = []
    for s in n_split:
        # data loading
        train_X = sc.read_h5ad(io_config["input_training_x"]).X
        n_cells, n_ftrs = train_X.shape
        sample_idx = np.random.choice(n_cells, n_cells // n_split)

        train_X = train_X[sample_idx, :].toarray()
        train_y = sc.read_h5ad(io_config["input_training_y"]).X[sample_idx, :].toarray()

        output_dir = io_config["output_dir"]
        output_dir = Path(output_dir / f"{model_config['model_name']}_{model_config['fs_name']}_{model_config['fs_est']}")

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
    test_X = sc.read_h5ad(io_config["input_test_x"])

    for i, clf in enumerate(cv_clfs):
        predicted_y = clf.predict(test_X.X.toarray())
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
    model_cfd = cfd / "models"

    parser.add_argument("-m", "--model", help="model config file name or location")
    parser.add_argument("-i", "--io", help="IO config file name or location", default="cite_io_cfg")
    parser.add_argument("-c", "--cv", help="cv config file name or location", default="cv_cfg")
    parser.add_argument("-n", "--num", help="Number of bootstrap samplings", default=5, type=int)

    args = parser.parse_args()
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
