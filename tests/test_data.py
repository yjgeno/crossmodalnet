import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from crossmodalnet.data import load_data, sc_Dataset


@pytest.fixture
def h5ad_pair(tmp_path):
    rng = np.random.default_rng(0)
    n_cells, n_genes, n_proteins = 40, 20, 5
    cell_ids = [f"c{i}" for i in range(n_cells)]

    X = csr_matrix(rng.poisson(1.0, size=(n_cells, n_genes)).astype(float))
    obs = pd.DataFrame(
        {"day": rng.choice([2, 3, 4, 7], size=n_cells)},
        index=cell_ids,
    )
    var_x = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    adata_x = ad.AnnData(X=X, obs=obs, var=var_x)

    Y = csr_matrix(rng.normal(0.0, 1.0, size=(n_cells, n_proteins)).astype(float))
    var_y = pd.DataFrame(index=[f"p{i}" for i in range(n_proteins)])
    adata_y = ad.AnnData(X=Y, obs=obs.copy(), var=var_y)

    px = tmp_path / "x.h5ad"
    py = tmp_path / "y.h5ad"
    adata_x.write_h5ad(px)
    adata_y.write_h5ad(py)
    return str(px), str(py), n_genes, n_proteins


def test_sc_dataset_shapes(h5ad_pair):
    px, py, n_genes, n_proteins = h5ad_pair
    ds = sc_Dataset(data_path_X=px, data_path_Y=py, time_key="day")
    assert ds.n_feature_Y == n_proteins
    x, day, y = ds[0]
    assert x.shape == (ds.n_feature_X,)
    assert y.shape == (n_proteins,)
    assert day.ndim == 1


def test_load_data_splits(h5ad_pair):
    px, py, _, _ = h5ad_pair
    ds = sc_Dataset(data_path_X=px, data_path_Y=py, time_key="day")
    train_loader, val_loader = load_data(ds, split=0.2, batch_size=4)
    n_train = sum(b[0].shape[0] for b in train_loader)
    n_val = sum(b[0].shape[0] for b in val_loader)
    assert n_train + n_val == len(ds)
    assert n_val > 0
