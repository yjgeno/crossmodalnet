import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import scipy
import numpy as np
from typing import Union
from sklearn.preprocessing import OneHotEncoder
from .utils import check_training_data, get_chrom_dicts, preprocessing_, Indexer


class sc_Dataset(Dataset):
    """
    Dataset class to load AnnData.
    """
    def __init__(self, 
                data_path_X, 
                data_path_Y,
                time_key: str = "day",
                celltype_key: str = "cell_type",
                preprocessing_key: str = None,
                **kwargs
                ):
        """
        Args:
            data_path: Path to .h5ad file.
        """
        data = sc.read_h5ad(data_path_X)
        sc.pp.filter_genes(data, min_cells = 5) # filter feature
        print(f"Features to use: {data.shape[1]}")
        counts = data.X.toarray() if scipy.sparse.issparse(data.X) else data.X # dense
        counts = preprocessing_(counts, key = preprocessing_key, **kwargs)
        data_Y = sc.read_h5ad(data_path_Y)
        check_training_data(data, data_Y)
        try:
            self.chrom_len_dict, self.chrom_idx_dict = get_chrom_dicts(data)
        except Exception:
            pass
        self.X = torch.Tensor(counts)
        self.var_names_X = data.var_names.to_numpy() # X feature names
        self.n_feature_X = counts.shape[1] # X feature no.
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
        self.n_feature_Y = len(self.var_names_Y)
        
        
    def __len__(self):
        return len(self.celltype)

    def __getitem__(self, idx):
        """
        return a tuple X, Y
        """
        return self.X[idx], self.day[idx], self.celltype_dict[self.celltype[idx]], self.Y[idx] # tuple


class sc_Dataset_index(Dataset):
    """
    Dataset class to index AnnData.
    """
    def __init__(self, 
                data_path_X, 
                data_path_Y,
                preprocessing_key = "binary",
                **kwargs
                ):
        """
        Args:
            data_path: Path to .h5ad file.
        """
        data = sc.read_h5ad(data_path_X)
        sc.pp.filter_genes(data, min_cells = 5) # filter feature
        print(f"Features to use: {data.shape[1]}")
        counts = data.X.toarray() if scipy.sparse.issparse(data.X) else data.X # dense
        self.X = preprocessing_(counts, key = preprocessing_key, **kwargs)   
        
        self.var_names_X = data.var_names.to_numpy() # selected X feature names
        self.n_feature_X = counts.shape[1] # X feature no.
#         self.indexer = Indexer()
#         for feature in data.var["gene_id"]:
#             self.indexer.add_and_get_index(feature)
        
        data_Y = sc.read_h5ad(data_path_Y)
        check_training_data(data, data_Y)      
        self.Y = (
            torch.Tensor(data_Y.X.A)
            if scipy.sparse.issparse(data_Y.X)
            else torch.Tensor(data_Y.X)
        )
        self.var_names_Y = data_Y.var_names.to_numpy() # Y feature names
        self.n_feature_Y = len(self.var_names_Y)
        
    def __len__(self):
        return len(self.celltype)

    def __getitem__(self, idx):
        """
        return a tuple X_indexed, Y
        """
        return torch.LongTensor(np.where(self.X[idx]!=0)[0]), self.Y[idx] 


class Collator(object):
    def __init__(self, max_len=None):
        self._max_len = max_len

    @property
    def max_len(self):
        return self._max_len

    def __call__(self, data):
        """
        data: a list of tuples (x_indexed, labels) for batching,
              where x_indexed is a tensor of arbitrary shape.
        """
        x_indexed, labels = zip(*data)
        if self.max_len is None:
            lengths = [x.shape[0] for x in x_indexed]
            self.max_len = max(lengths) # max_len varies across batches
        # print("max_len", max_len)
        # features = torch.zeros((len(data), max_len))
        x_indexed_padd = torch.LongTensor([-1]).repeat(len(data)*self.max_len).view(len(data), self.max_len)
        for i in range(len(data)):
            x_indexed_padd[i][:len(data[i][0])] = data[i][0]

        return x_indexed_padd.long(), torch.stack(labels)


def load_data(dataset: Union[sc_Dataset, sc_Dataset_index],
              split: float = 0.15, # train/val
              batch_size: int = 256, 
              shuffle: bool = True, 
              random_state: int = 0,
              **kwargs
              ):
    n_val = int(split * len(dataset))
    torch.manual_seed(random_state)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-n_val, n_val])
    print(f"split data: Train/Val = {1-split}/{split}")
    return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **kwargs), DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, **kwargs)
