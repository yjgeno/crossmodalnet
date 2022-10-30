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

    # prediction
    del train_X
    del train_y
    test_X = sc.read_h5ad(io_config["input_test_x"])
    predicted_y = cv_object.predict(test_X.X.toarray())

    # save result
    pd.DataFrame(data=predicted_y).to_csv(output_dir / "predicted_y.csv")
    cv_object.save_iters(output_dir / "cv_result.csv")
    cv_object.save_model(output_dir / "best_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--io", help="IO config file location")
    parser.add_argument("-m", "--model", help="model config file location")
    parser.add_argument("-c", "--cv", help="cv config file location")
    parser.add_argument("-n", "--num", help="Number of bootstrap samplings")

    args = parser.parse_args()
    model_configs = load_config(args.model)
    model_name = model_configs.pop("model")
    fs_name = model_configs.pop("fs_name")
    fs_est = model_configs.pop("fs_estimator")
    train(io_config=load_config(args.io),
          model_config={"model_name": model_name,
                        "fs_name": fs_name,
                        "fs_est": fs_est,
                        "param": parse_config(model_configs)},
          cv_config=load_config(args.cv))
