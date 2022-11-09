import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from anndata import AnnData 


def get_chrom_dicts(ada: AnnData):
    """
    Get two dictionaries for each chromosome's feature len and feature start/end index.
    """
    df_meta = ada.var["gene_id"].str.split(':', expand=True).rename(columns={0:"chr", 1:'region'})
    df_meta.reset_index(inplace=True)
    chrom_len_dict = (df_meta["chr"].value_counts()[:23]).drop(labels="chrX").to_dict() # chr1-chr22, w/o chrX, chrY
    chrom_idx_dict = {}
    for chrom in chrom_len_dict.keys(): # same key order
        chrom_idx_dict[chrom] = (df_meta.groupby("chr").groups[chrom]).to_numpy()[[0,-1]].tolist()
    return chrom_len_dict, chrom_idx_dict


def preprocessing_(counts, key: str = None, **kwargs):
    """
    Args:
        counts (dense array): cell * feature.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn import decomposition  
    # min_cells = 5
    # counts = counts[:, np.asarray(np.sum(counts>0, axis=0) > min_cells).ravel()]
    # print(f"Features more than {min_cells} cells to use: {counts.shape[1]}")
    scaler = StandardScaler()

    if key == "binary":
        counts[counts != 0] = 1.
    if key == "standard_0":       
        counts = scaler.fit_transform(counts) # default on features
    if key == "standard_1":
        counts = scaler.fit_transform(counts.T).T # on sample
    if key == "PCA":
        counts = scaler.fit_transform(counts) # on features
        pca = decomposition.PCA(n_components = 50, svd_solver="full", **kwargs)
        counts = pca.fit_transform(counts)
    if key == "tSVD": # TruncatedSVD on TF/IDF data
        svd = decomposition.TruncatedSVD(n_components = 50, n_iter=7, random_state=42, **kwargs)
        counts = svd.fit_transform(counts)
    if key == "magic":
        # pip install --user magic-impute 
        import magic
        magic_op = magic.MAGIC(knn=7)
        counts = magic_op.fit_transform(counts)
    print(f"Complete preprocessing by {key}")

    return counts


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


class Indexer(object):
    """
    Bijection between objects and integers starting at 0.
    labels, features, etc. into coordinates of a vector space.
    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        return self.index_of(object) != -1

    def index_of(self, object):
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints) # add 1 len as index
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


def aggregate_bin(data,
                  var,
                  chr_name,
                  distance=10000,
                  split=None,
                  inclusive="both",
                  agg_method="sum"):
    start_loc, end_loc = var["loc_start"].min(), var["loc_end"].max()
    matrixes, vars = [], []
    bin_range = distance if split is None else ((end_loc - start_loc) / split)
    for s in range(start_loc, end_loc, bin_range):
        t = s + bin_range
        if inclusive == "both":
            sel_vars = var[(var["loc_start"] < t) &
                           (var["loc_end"] > s)]
        elif inclusive == "left":
            sel_vars = var[(var["loc_start"] < t) &
                           (t >= var["loc_end"] > s)]
        elif inclusive == "right":
            sel_vars = var[(s <= var["loc_start"] < t) &
                           (var["loc_end"] > s)]
        else:
            sel_vars = var[(s <= var["loc_start"] < t) &
                           (t >= var["loc_end"] > s)]
        if data[:, sel_vars.index].X.shape[1] == 0:
            continue
        matrixes.append(getattr(data[:, sel_vars.index].X, agg_method)(axis=1))
        vars.append(f"{chr_name}_{s}:{t}")

    return np.concatenate(matrixes, axis=1), pd.DataFrame(vars, index=vars)


def bin_var(data: AnnData,
            distance: int = 100000,
            split: int = None,  # number of splits
            inclusive="both"):
    var = data.var.copy()
    var["chromosome"] = var["gene_id"].apply(lambda x: x.split(":")[0])
    var["gene_loc"] = var["gene_id"].apply(lambda x: x.split(":")[1])
    var["loc_start"] = var["gene_loc"].apply(lambda x: int(x.split("-")[0]))
    var["loc_end"] = var["gene_loc"].apply(lambda x: int(x.split("-")[1]))
    del var["gene_loc"]
    matrixes, vars = [], []
    for ch in tqdm(var["chromosome"].unique()):
        matrix, var_names = aggregate_bin(data[:, var[var["chromosome"] == ch].index],
                                          var[var["chromosome"] == ch],
                                          ch, distance, split, inclusive)
        matrixes.append(matrix)
        vars.append(var_names)
    adata = AnnData(X=np.concatenate(matrixes, axis=1),
                    obs=data.obs,
                    var=pd.concat(vars, axis=0))
    adata.var.rename(columns={0:"gene_id"}, inplace=True)
    return adata