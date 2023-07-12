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
    
    if args.eval_time:
        X_train, y_train = sc.read_h5ad(data_configs["X_train_pth"]), sc.read_h5ad(data_configs["y_train_pth"])
        X_train, y_train = get_subset(X_train, y_train,
                                      n_obs=int(args.obs),
                                      n_vars=int(args.var))
        if args.eval_memory:
            with open(Path(data_configs["result_dir_pth"]) / data_configs["mem_file_name"], "w+") as fp:
                est_time = train_eval_time_mem(eval_time, fp)(adata_X=X_train, y_train=y_train)
            est_mem = extract_peak_mem(Path(data_configs["result_dir_pth"]) / data_configs["mem_file_name"])
        else:
            est_time = eval_time(adata_X=X_train, y_train=y_train)
            est_mem = None
        result_dir = Path(data_configs["result_dir_pth"]).mkdir(parents=True, exist_ok=True)
        print_info(est_time, n_obs=int(args.obs), n_var=int(args.var), hparams={}, memory_used=est_mem,
                   file_name= Path(data_configs["result_dir_pth"]) / f"totalVI.csv")
    else:
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