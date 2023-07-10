import subprocess
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import mmread
import scanpy as sc
from anndata import AnnData


class cTPnetDataset(Dataset):
    def __init__(self, 
                 X_path, 
                 y_path):
        X_files = ["denoised_var.csv", "denoised_obs.csv", "denoised_x.mtx"]
        for x_file in X_files:
            if not (Path(X_path) / x_file).is_file():
                raise FileNotFoundError(x_file, f"is not in the {X_path}")
        self.y = sc.read_h5ad(y_path)
        self.x_obs = pd.read_csv(Path(X_path) / "denoised_obs.csv", index_col=0)
        self.x_var = pd.read_csv(Path(X_path) / "denoised_var.csv", index_col=0)
        self.x_obs.index = self.x_obs["x"]
        self.x_var.index = self.x_var["x"]

        if len(set(self.x_obs.index) ^ set(self.y.obs.index)) != 0:
            raise KeyError("Non-overlapped observations found:", 
                           set(self.x_obs.index) ^ set(self.y.obs.index))

        with open(Path(X_path) / "denoised_x.mtx") as f:
            self.X = AnnData(X=mmread(f).T, var=self.x_var, obs=self.x_obs)

    def _normalize(self, x):
        return np.log(x * 1e6 / x.sum(axis=0) + 1)
    
    @property
    def n_proteins(self):
        return self.y.var.shape[0]
    
    @property
    def n_genes(self):
        return self.X.var.shape[0]
    
    def __len__(self):
        return self.X.obs.shape[0]
    
    def __getitem__(self, idx):
        return {"X": self._normalize(self.X.X.toarray()[idx, :].astype(float)),
                "y": self.y[idx, :].X.toarray().astype(float)}  # we don't normalize y here since it's normalized values already


def savexr(data_path, output_path, pretrained):
    subprocess.run(["Rscript", "./baselines/cTPnet/denoise.R", 
                    data_path, output_path, pretrained])
