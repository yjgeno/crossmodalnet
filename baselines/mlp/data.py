from torch.utils.data import Dataset
import scanpy as sc


class scDataset(Dataset):
    def __init__(self,
                 X_path,
                 y_path,
                 X_transform=lambda x: x):
        if X_path is not None:
            self.X = sc.read_h5ad(X_path)
            print("loaded X with obs: ", self.X.obs.columns)
        if y_path is not None:
            self.y = sc.read_h5ad(y_path)
        self.X_transform = X_transform

    @property
    def n_proteins(self):
        return self.y.var.shape[0]

    @property
    def n_genes(self):
        return self.X.var.shape[0]

    @classmethod
    def init_with_data(cls, X, y):
        new_ = cls(None, None)
        new_.X = X
        new_.y = y
        return new_

    def __len__(self):
        return self.X.obs.shape[0]

    def __getitem__(self, idx):
        return {"X": self.X_transform(self.X[idx, :].X.toarray().astype(float)),
                "y": self.y[idx, :].X.toarray().astype(float)}