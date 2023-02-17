import numpy as np
from anndata import AnnData 
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import warnings


def sc_preprocess(adata: AnnData, 
                  normalize = True, 
                  log = True,
                  pseudocount = 1,
                  scale = True, 
                  **kwargs
                  ):
    # adata.X.A = np.clip(adata.X.A, 0, 100)
    if normalize:
        sc.pp.normalize_total(adata, target_sum = 1e4, **kwargs)
        print("Cell-wise normalize AnnData")
    if log:
        # sc.pp.log1p(adata, **kwargs)
        adata.X.A = np.log(adata.X.A + pseudocount)
        print("Logarithmize AnnData")
    if scale:
        sc.pp.scale(adata, **kwargs)
        print("Center features of AnnData")


class pretransformer:
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
        
        print(f"Transform counts by {self.key}")
        return counts


def check_training_data(ada_X: AnnData, ada_Y: AnnData):
    assert (ada_X.obs.index == ada_Y.obs.index).all() # match cells
    # if not ("day" and "cell_type") in ada_X.obs.columns:
    #     raise ValueError("")


def split_data(adata, split=0.15, cell_id_test=None, random_state=0, time_key="day"):
    """
    Split data into train/val and test.
    """
    if cell_id_test is None:
        np.random.seed(random_state)
        df = adata.obs.groupby(time_key).apply(lambda x: x.sample(int(split*len(x)))) #.set_index("cell_id")
        cell_id_test = [i[1] for i in df.index.to_numpy()]
        np.savetxt("cell_id_test.txt", cell_id_test, delimiter="\t", fmt="%s") 
    adata_test = adata[adata.obs.index.isin(cell_id_test), :].copy()
    adata_train = adata[~adata.obs.index.isin(cell_id_test), :].copy()
    return adata_train, adata_test


def corr_score(y_true, y_pred):
    """
    Returns the average of each sample's Pearson correlation coefficient.
    """
    if np.std(y_true)==0 or np.std(y_pred)==0 :
        warnings.warn("Standard deviation equals to zero")
    corrsum = 0
    for i in range(len(y_true)): # aggregate samples in a batch
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)
