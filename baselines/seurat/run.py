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
                           y_pred_pth):
    converter = H5adToMtx(X_path=X_train_pth)
    converter.run(save_path=X_train_p_pth, transport=False)

    converter = H5adToMtx(X_path=y_train_pth)
    converter.run(save_path=y_train_p_pth, transport=False)

    converter = H5adToMtx(X_path=X_test_pth)
    converter.run(save_path=X_test_p_pth, transport=False)

    subprocess.run(["Rscript", "./baselines/seurat/seurat_integration.R", 
                    X_train_p_pth, y_train_p_pth, X_test_p_pth, y_pred_pth])
    

def evaluate(y_pred_pth, y_true_pth, result_df_pth):
    y_true = sc.read_h5ad(y_true_pth)
    y_pred = load_mtx_dir(mtx=Path(y_true_pth) / "pred_x.mtx",
                 var=Path(y_true_pth) / "var.csv",
                 obs=Path(y_true_pth) / "obs.csv")
    
    


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