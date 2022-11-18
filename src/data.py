import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import scipy
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from .utils import check_training_data, preprocessor


class sc_Dataset(Dataset):

    def __init__(self, 
                data_path_X, 
                data_path_Y,
                time_key: str = "day",
                celltype_key: str = "cell_type",
                preprocessing_key: str = None,
                save_prep: bool = False,
                **kwargs
                ):
        """

        Args:
            data_path: Path to .h5ad file.
        """
        # process X
        data = sc.read_h5ad(data_path_X)
        sc.pp.filter_genes(data, min_cells = 5) # filter feature
        print(f"Features to use: {data.shape[1]}")
        counts = data.X.toarray() if scipy.sparse.issparse(data.X) else data.X # dense
        self.processor = preprocessor(key = preprocessing_key)
        counts = self.processor(counts, **kwargs)
        if save_prep:
            np.save("x_selected_features_.npy", data.var.index.to_numpy())
        # print("components_", self.processor.svd.components_.shape) # [#PCs, Features to use]
        self.X = torch.Tensor(counts)
        self.var_names_X = data.var_names.to_numpy() # X feature names
        self.n_feature_X = counts.shape[1] # init n_input

        # process meta data
        self.day = data.obs[time_key].to_numpy()
        self.unique_day = np.sort(np.unique(data.obs[time_key].values)) # [2,3,4,7]
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

        # process Y
        data_Y = sc.read_h5ad(data_path_Y)
        check_training_data(data, data_Y)
        counts_Y = data_Y.X.toarray() if scipy.sparse.issparse(data_Y.X) else data_Y.X # dense
        self.n_feature_Y = counts_Y.shape[1] # init n_output 
        self.Y = torch.Tensor(counts_Y)
        self.var_names_Y = data_Y.var_names.to_numpy() # Y feature names
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        return a tuple X, Y
        """
        return self.X[idx], self.day_dict[self.day[idx]], self.celltype_dict[self.celltype[idx]], self.Y[idx] # tuple


def load_data(dataset: sc_Dataset,
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
