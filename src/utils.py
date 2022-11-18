import numpy as np
from anndata import AnnData 
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition


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
    # if not ("day" and "cell_type") in ada_X.obs.columns:
    #     raise ValueError("")


def corr_score(y_true, y_pred):
    """
    Returns the average of each sample's Pearson correlation coefficient.
    """
    corrsum = 0
    for i in range(len(y_true)): # aggregate samples in a batch
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)
