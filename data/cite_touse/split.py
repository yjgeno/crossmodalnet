import numpy as np
import pandas as pd 
import scanpy as sc

def split_data(adata, split=0.15, cell_id_test=None, random_state=0, time_key="day", return_cell_id=False):
    if cell_id_test is None:
        np.random.seed(random_state)
        df = adata.obs.groupby(time_key).apply(lambda x: x.sample(int(split*len(x)))) #.set_index("cell_id")
        cell_id_test = [i[1] for i in df.index.to_numpy()]
        np.savetxt("cell_id_test.txt", cell_id_test, delimiter="\t", fmt="%s") 
    adata_test = adata[adata.obs.index.isin(cell_id_test), :].copy()
    adata_train = adata[~adata.obs.index.isin(cell_id_test), :].copy()
    if return_cell_id is True:
    	return adata_train, adata_test, cell_id_test
    return adata_train, adata_test

ada_x = sc.read_h5ad("/scratch/user/yjyang027/open-problems-multimodal/cite_train_x.h5ad")
ada_y = sc.read_h5ad("/scratch/user/yjyang027/open-problems-multimodal/cite_train_y.h5ad")
print(ada_x.shape, ada_y.shape)

ada_train_val_x, ada_test_x, cell_id_test = split_data(ada_x, split=0.15, return_cell_id=True)
print(ada_train_val_x.shape, ada_test_x.shape)
for ada in [ada_train_val_x, ada_test_x]:
    print(ada.obs["day"].value_counts())
ada_train_val_x.write_h5ad("cite_train_val_x.h5ad")
ada_test_x.write_h5ad("cite_test_x.h5ad")

ada_train_val_y, ada_test_y = split_data(ada_y, cell_id_test=cell_id_test)
print(ada_train_val_y.shape, ada_test_y.shape)
for ada in [ada_train_val_y, ada_test_y]:
    print(ada.obs["day"].value_counts())
ada_train_val_y.write_h5ad("cite_train_val_y.h5ad")
ada_test_y.write_h5ad("cite_test_y.h5ad")
