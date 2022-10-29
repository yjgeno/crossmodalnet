import numpy as np
import torch
from anndata import AnnData 


def get_chrom_dicts(ada):
    """
    Get two dictionaries for each chromosome's feature len and feature start/end index.
    """
    df_meta = ada.var["gene_id"].str.split(':', expand=True).rename(columns={0:"chr", 1:'region'})
    df_meta.reset_index(inplace=True)
    chrom_len_dict = (df_meta["chr"].value_counts()[:23]).to_dict() # chr1-chr22, chrX, w/o chrY
    chrom_idx_dict = {}
    for chrom in chrom_len_dict.keys(): # same key order
        chrom_idx_dict[chrom] = (df_meta.groupby("chr").groups[chrom]).to_numpy()[[0,-1]].tolist()
    return chrom_len_dict, chrom_idx_dict


def check_training_data(ada_X: AnnData, ada_Y: AnnData):
    assert (ada_X.obs.index == ada_Y.obs.index).all() # match cells
    if not ("day" and "cell_type") in ada_X.obs.columns:
        raise ValueError("")


def test_to_tensor(ada: AnnData):
    """
    Convert adata counts to tensor for model input.
    """
    test_x = ada.X.tocoo()
    values = test_x.data
    indices = np.vstack((test_x.row, test_x.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = test_x.shape
    test_x = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
    return test_x


def corr_score(y_true, y_pred):
    """
    Returns the average of each sample's Pearson correlation coefficient.
    """
    corrsum = 0
    for i in range(len(y_true)): # aggregate samples in a batch
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)
