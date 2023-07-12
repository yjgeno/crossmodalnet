import os
os.environ["OMP_NUM_THREADS"] = '1'

import matplotlib.pyplot as plt
import scanpy as sc
# import scvi
import umap
import pandas as pd
import numpy as np
from scvi.model import TOTALVI
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from anndata import AnnData
from time import time


def eval_time(adata_X, 
              y_train):
    adata_X.obsm["protein_expression"] = pd.DataFrame(data=y_train.X.toarray(), index=y_train.obs.index, columns=y_train.var.index)
    del y_train

    adata_X.layers["counts"] = np.expm1(adata_X.X)
    adata_X.layers["counts"] = adata_X.layers["counts"].astype(int)
    adata_X.raw = adata_X
    TOTALVI.setup_anndata(
        adata_X,
        layer="counts",
        protein_expression_obsm_key="protein_expression",
    )

    arches_params = dict(
        use_layer_norm="both",
        use_batch_norm="none",
        n_layers_decoder=2,
        n_layers_encoder=2,
    )
    start_time = time()
    vae = TOTALVI(adata_X, **arches_params)
    vae.train(max_epochs=200)
    return time() - start_time



def make_data(x_train_path,
              y_train_path):
    adata_X = sc.read_h5ad(x_train_path)
    adata_y = sc.read_h5ad(y_train_path)
    adata_X.obsm["protein_expression"] = pd.DataFrame(data=adata_y.X.toarray(), index=adata_y.obs.index, columns=adata_y.var.index)
    del adata_y

    adata_X.layers["counts"] = np.expm1(adata_X.X)
    adata_X.layers["counts"] = adata_X.layers["counts"].astype(int)
    adata_X.raw = adata_X
    return adata_X


def train_totalVI(x_train_path,
                y_train_path,
                model_path="seurat_reference_model",
                batch_key="week"):
    adata_X = make_data(x_train_path,
                        y_train_path)

    TOTALVI.setup_anndata(
        adata_X,
        layer="counts",
        batch_key=batch_key,
        protein_expression_obsm_key="protein_expression",
    )

    arches_params = dict(
        use_layer_norm="both",
        use_batch_norm="none",
        n_layers_decoder=2,
        n_layers_encoder=2,
    )
    vae = TOTALVI(adata_X, **arches_params)
    vae.train(max_epochs=200)
    vae.save(model_path, overwrite=True)
    return adata_X


def inference(adata_X, 
              x_test_path,
              y_test_path,
              y_pred_save_path,
              model_path="seurat_reference_model"):
    vae = TOTALVI.load(model_path, adata=adata_X)

    query = sc.read_h5ad(x_test_path)
    query_y = sc.read_h5ad(y_test_path)
    query.obsm["protein_expression"] = pd.DataFrame(data=np.zeros(shape=(query_y.obs.index.shape[0],
                                                                        query_y.var.index.shape[0])), 
                                                                        index=query_y.obs.index, 
                                                                        columns=query_y.var.index)
    del query_y

    query.layers["counts"] = np.expm1(query.X)
    query.layers["counts"] = query.layers["counts"].astype(int)
    query.raw = query
    query.obsm["protein_expression"] = query.obsm["protein_expression"].loc[
        :, adata_X.obsm["protein_expression"].columns
    ]
    adata_X.obs["dataset_name"] = "Reference"
    query.obs["dataset_name"] = "Query"
    query = query[:, adata_X.var_names].copy()
    vae_q = TOTALVI.load_query_data(
        query,
        vae,
        freeze_expression=True
    )
    vae_q.train(
        max_epochs=100,
        plan_kwargs=dict(weight_decay=0.0, scale_adversarial_loss=0.0),
    )
    _, predicted_protein = vae_q.get_normalized_expression()
    predicted_protein.to_csv(y_pred_save_path)


def calc_metrics(y_pred_save_path,
                 y_test_path,
                 result_df_pth):
    y_true = sc.read_h5ad(y_test_path)
    sc.pp.normalize_total(y_true)
    sc.pp.log1p(y_true)

    y_pred = AnnData(pd.read_csv(y_pred_save_path, index_col=0))
    sc.pp.normalize_total(y_pred)
    sc.pp.log1p(y_pred)

    results = {}
    prs = [stats.pearsonr(y_true.X.toarray()[:, i], 
                          y_pred.X[:, i])[0] for i in range(y_true.var.shape[0])]
    results["PearsonR_mean"] = np.mean(prs)
    results["PearsonR_std"] = np.std(prs)

    r2s = r2_score(y_true.X.toarray(), y_pred.X, multioutput="raw_values")
    results["R2_mean"] = np.mean(r2s)
    results["R2_std"] = np.std(r2s)

    mses = mean_squared_error(y_true.X.toarray(), y_pred.X, multioutput="raw_values")
    results["MSE_mean"] = np.mean(mses)
    results["MSE_std"] = np.std(mses)

    pd.DataFrame({"score": results}).to_csv(result_df_pth)