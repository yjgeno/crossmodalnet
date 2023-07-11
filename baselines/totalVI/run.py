import os
os.environ["OMP_NUM_THREADS"] = '1'
import argparse
import subprocess
import scanpy as sc
from baselines.utils import *
from baselines.totalVI.model import *




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--io", help="Path to io configs", default='io')
    parser.add_argument("-o", "--obs", help="Number of observations", default='300')
    parser.add_argument("-v", "--var", help="Number of variables", default='1000')
    parser.add_argument("-e", "--eval_time", help="To evaluate training time or not",
                        action='store_true')
    parser.add_argument("-m", "--eval_memory", help="To evaluate memory usage or not",
                        action='store_true')

    args = parser.parse_args()
    cfd = Path(__file__).resolve().parent.parent / "configs"
    data_configs = load_config((cfd / args.io).with_suffix(".yml")) \
        if (cfd / args.io).with_suffix(".yml").is_file() else load_config(args.io)

    adata_x = train_totalVI(x_train_path=data_configs["X_train_pth"],
                            y_train_path=data_configs["y_train_pth"],
                            model_path=data_configs["model_pth"],
                            batch_key=data_configs["batch_key"])
    
    inference(adata_X=adata_x, 
              x_test_path=data_configs["X_test_pth"],
              y_test_path=data_configs["y_test_pth"],
              y_pred_save_path=Path(data_configs["result_dir_pth"])/"y_pred.csv",
              model_path=data_configs["model_pth"])
    
    calc_metrics(y_pred_save_path=Path(data_configs["result_dir_pth"])/"y_pred.csv",
                 y_test_path=data_configs["y_test_pth"],
                 result_df_pth=Path(data_configs["result_dir_pth"])/"result.csv")