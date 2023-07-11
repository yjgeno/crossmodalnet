import argparse
import subprocess
import scanpy as sc
from baselines.utils import *
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats


def run_seurat_ref_mapping(X_train_pth,
                           X_train_p_pth,
                           y_train_pth,
                           y_train_p_pth,
                           X_test_pth,
                           X_test_p_pth,
                           y_pred_pth,
                           **kwargs):
    converter = H5adToMtx(X_path=X_train_pth)
    converter.run(save_path=X_train_p_pth, transport=True)

    converter = H5adToMtx(X_path=y_train_pth)
    converter.run(save_path=y_train_p_pth, transport=True)

    converter = H5adToMtx(X_path=X_test_pth)
    converter.run(save_path=X_test_p_pth, transport=True)

    subprocess.run(["Rscript", "./baselines/seurat/seurat_integration.R", 
                    X_train_p_pth, y_train_p_pth, X_test_p_pth, y_pred_pth])
    

def evaluate(y_pred_pth, y_true_pth, result_df_pth):
    y_true = sc.read_h5ad(y_true_pth)
    y_pred = load_mtx_dir(mtx=Path(y_pred_pth) / "pred_x.mtx",
                         var=Path(y_pred_pth) / "var.csv",
                         obs=Path(y_pred_pth) / "obs.csv")
    
    results = {}
    prs = [stats.pearsonr(y_true.X.toarray()[:, i], 
                          y_pred.X.toarray()[:, i])[0] for i in range(y_true.var.shape[0])]
    results["PearsonR_mean"] = np.mean(prs)
    results["PearsonR_std"] = np.std(prs)

    r2s = r2_score(y_true.X.toarray(), y_pred.X.toarray(), multioutput="raw_values")
    results["R2_mean"] = np.mean(r2s)
    results["R2_std"] = np.std(r2s)

    mses = mean_squared_error(y_true.X.toarray(), y_pred.X.toarray(), multioutput="raw_values")
    results["MSE_mean"] = np.mean(mses)
    results["MSE_std"] = np.std(mses)

    pd.DataFrame({"score": results}).to_csv(result_df_pth)


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

    run_seurat_ref_mapping(X_train_pth=data_configs["X_train_pth"],
                           X_train_p_pth=data_configs["X_train_processed_dir"],
                           y_train_pth=data_configs["y_train_pth"],
                           y_train_p_pth=data_configs["y_train_processed_dir"],
                           X_test_pth=data_configs["X_test_pth"],
                           X_test_p_pth=data_configs["X_test_processed_dir"],
                           y_pred_pth=data_configs["result_dir_pth"])
    
    evaluate(y_pred_pth=data_configs["result_dir_pth"], 
             y_true_pth=data_configs["y_test_pth"], 
             result_df_pth=Path(data_configs["result_dir_pth"])/"result.csv")