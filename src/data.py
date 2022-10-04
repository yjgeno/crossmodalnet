import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import scipy
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from .utils import check_training_data


class sc_Dataset(Dataset):

    def __init__(self, 
                data_path_X, 
                data_path_Y,
                time_key: str = "day",
                celltype_key: str = "cell_type",
                ):
        """

        Args:
            data_path: Path to .h5ad file.
        """
        data = sc.read_h5ad(data_path_X)
        data_Y = sc.read_h5ad(data_path_Y)
        check_training_data(data, data_Y)
        self.X = (
            torch.Tensor(data.X.A)
            if scipy.sparse.issparse(data.X)
            else torch.Tensor(data.X)
        )
        self.var_names = data.var_names.to_numpy() # X feature names
        self.day = data.obs[time_key].to_numpy()
        self.unique_day = np.unique(data.obs[time_key].values)
        self.celltype = data.obs[celltype_key].to_numpy()
        self.unique_celltype = np.unique(data.obs[celltype_key].values)

        encoder_day = OneHotEncoder(sparse=False)
        encoder_day.fit(self.unique_day.reshape(-1, 1))  # (# day, 1)
        # self.encoder_day = encoder_day
        self.day_dict = dict(
                zip(
                    self.unique_day,
                    torch.Tensor(encoder_day.transform(self.unique_day.reshape(-1, 1))),
                )
            )     
        encoder_celltype = OneHotEncoder(sparse=False)
        encoder_celltype.fit(self.unique_celltype.reshape(-1, 1))  # (# celltype, 1)
        # self.encoder_celltype = encoder_celltype
        self.celltype_dict = dict(
                zip(
                    self.unique_celltype,
                    torch.Tensor(encoder_celltype.transform(self.unique_celltype.reshape(-1, 1))),
                )
            )
        
        self.Y = (
            torch.Tensor(data_Y.X.A)
            if scipy.sparse.issparse(data_Y.X)
            else torch.Tensor(data_Y.X)
        )
        self.var_names_Y = data_Y.var_names.to_numpy() # Y feature names
        
        
    def __len__(self):
        return len(self.celltype)

    def __getitem__(self, idx):
        """
        return a tuple X, Y
        """
        return self.X[idx], self.day[idx], self.celltype_dict[self.celltype[idx]], self.Y[idx] # tuple


def load_data(data_path_X, 
              data_path_Y, 
              split: float = 0.25, # train/val
              batch_size: int = 128, 
              shuffle: bool = False, 
              **kwargs
              ):
    dataset = sc_Dataset(data_path_X, data_path_Y)
    n_val = int(split * len(dataset))
    torch.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-n_val, n_val])
    return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **kwargs), DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, **kwargs)
