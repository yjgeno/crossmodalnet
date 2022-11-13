import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from anndata import AnnData 
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

sc.settings.verbosity = 0

CHROMS_IN_USE = [f"chr{i+1}" for i in range(22)]

def get_chrom_dicts(ada: AnnData):
    """
    Get two dictionaries for each chromosome's feature len and feature start/end index.
    """
    df_meta = ada.var["gene_id"].str.split(':', expand=True).rename(columns={0:"chr", 1:'region'})
    df_meta.reset_index(inplace=True)
    chrom_len_dict = (df_meta["chr"].value_counts()).to_dict() 
    chrom_len_dict = {chrom: chrom_len_dict[chrom] for chrom in CHROMS_IN_USE} # chr1-chr22, w/o chrX, chrY
    chrom_idx_dict = {}
    for chrom in chrom_len_dict.keys(): # same key order
        chrom_idx_dict[chrom] = (df_meta.groupby("chr").groups[chrom]).to_numpy()[[0,-1]].tolist()
    return chrom_len_dict, chrom_idx_dict


def reorder_chroms(ada: AnnData):
    """
    Reordering ATAC features according to chromosome locations.
    """
    df_meta = ada.var["gene_id"].str.split(':', expand=True).rename(columns={0:"chr", 1:'region'})
    df_meta = df_meta.join(df_meta["region"].str.split('-', expand=True).rename(columns={0:"start", 1:'end'}))
    df_meta.reset_index(inplace=True)
    df_meta_to_use = df_meta[df_meta["chr"].isin(CHROMS_IN_USE)]
    df_meta_to_use["chr_no"] = df_meta_to_use["chr"].str[3:] # "chr"
    df_meta_to_use.sort_values(by=['chr', 'start'], ascending=True, inplace=True)

    counts_new = np.zeros((len(df_meta_to_use), len(ada))) # [feature, cell]
    counts = ada.X.A.T
    for i, ii in enumerate(df_meta_to_use.index):
        counts_new[i] = counts[int(ii)] # reorder 
    return sc.AnnData(counts_new.T, obs=ada.obs, var=df_meta_to_use)


class preprocessor:
    def __init__(self, key = None):
        self.key = key
                
    def __call__(self, counts, **kwargs): # n_components
        """
        Args:
            counts (dense array): cell * feature.
        """
        if self.key == "binary":
            counts[counts != 0] = 1.
        if self.key == "standard_0":  
            self.scaler = StandardScaler()
            counts = self.scaler.fit_transform(counts) # default on features
        if self.key == "standard_1":
            self.scaler = StandardScaler()
            counts = self.scaler.fit_transform(counts.T).T # on sample
        if self.key == "PCA":
            self.scaler = StandardScaler()
            counts = self.scaler.fit_transform(counts) # on features
            self.pca = decomposition.PCA(svd_solver="full", **kwargs)
            counts = self.pca.fit_transform(counts)
        if self.key == "tSVD": # TruncatedSVD on TF/IDF data
            self.svd = decomposition.TruncatedSVD(n_iter=7, random_state=42, **kwargs)
            counts = self.svd.fit_transform(counts)
        if self.key == "magic":
            # pip install --user magic-impute 
            import magic
            self.magic_op = magic.MAGIC(knn=7, **kwargs)
            counts = self.magic_op.fit_transform(counts)
        if self.key == "tfidf":
            from sklearn.feature_extraction.text import TfidfTransformer
            self.tfidf = TfidfTransformer()
            counts = self.tfidf.fit_transform(counts).toarray()
        
        print(f"Complete preprocessing by {self.key}")
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
        vars.append(f"{chr_name}:{s}_{t}")

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