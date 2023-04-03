from os import path
import numpy as np
import pandas as pd
from anndata import AnnData 
import scanpy as sc
from scipy.sparse import csr_matrix
import torch
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mycolorpy import colorlist as mcp
import warnings

COLORS = mcp.gen_color(cmap="tab10", n=10)

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


def to_h5ad(df: pd.DataFrame, df_cell: pd.DataFrame,):
    adata = sc.AnnData(csr_matrix(df.to_numpy()))
    adata.var = pd.DataFrame(df.columns, columns=["gene_id"])
    adata.obs_names = df.index
    adata.var_names = adata.var['gene_id']
    adata.obs = pd.merge(adata.obs, df_cell, on=['cell_id']).set_index("cell_id")
    return adata


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


class saliency:
    def __init__(self,
                 counts: torch.Tensor, 
                 times: torch.Tensor,
                 model,
                 genes: list, 
                 proteins: list, 
                 ):
        if not (isinstance(genes, list) and isinstance(proteins, list)):
            print("Require list of genes and proteins")
        self.genes = genes
        self.proteins = proteins
        self.counts = counts
        self.times = times
        self.model = model
        self.set_TF()

    def set_TF(self):
        data_dir = path.join(path.dirname(path.dirname(path.abspath(__file__))), "data")
        self.TF = np.genfromtxt(path.join(data_dir, "TF.csv"), delimiter="\t", dtype=str)[1:]
        self.TF_intersect = np.array(list(set(self.genes).intersection(set(self.TF))))
        self.TF_intersect_idx = np.array([self.genes.index(t) for t in self.TF_intersect])

    def compute_saliency(self, 
                         protein_name: str,
                         normalize = True,
                         ):
        protein_j_saliency = []
        j = self.proteins.index(protein_name) # predict protein j
        for i, count in enumerate(self.counts): # loop across cells
            genes_i = count.unsqueeze(0)
            genes_i.requires_grad_()
            time_i = self.times[i].unsqueeze(0)
            pred_cite_yi = self.model(genes_i, T=time_i)
            protein_j = pred_cite_yi[:, j]
            protein_j.backward() # compute dj/di
            protein_j_saliency.append(genes_i.grad.data.abs())
        self.protein_j_saliency = torch.cat(protein_j_saliency)  
        if normalize: # min-max
              self.protein_j_saliency = self.protein_j_saliency/(self.protein_j_saliency.max()-self.protein_j_saliency.min())
        self.protein_j_saliency_mean, self.protein_j_saliency_std = self.protein_j_saliency.mean(dim=0), self.protein_j_saliency.std(dim=0)      
        print(f"Saliency of protein {protein_name} w.r.t. genes has been computed")
    
    def get_top_genes(self, 
                      k: int = 100, 
                      include_TF: bool = False,
                      ):
        self.top_genes, self.top_TFs = None, None
        try:
            _, self.idx_j = torch.topk(self.protein_j_saliency_mean, k)
        except:
            print("Compute saliency mean first")
        self.top_genes = np.array(self.genes)[self.idx_j]
        print(f"Select top {k} genes by saliency")
        if include_TF:
            _, self.idx_j_TF = torch.topk(self.protein_j_saliency_mean[self.TF_intersect_idx], min(k, len(self.TF_intersect_idx)))
            self.top_TFs = self.TF_intersect[self.idx_j_TF]
            print(f"Select top {k} TFs by saliency")

    @staticmethod
    def plot_hbar_(ys: np.array, 
                   yerrs: np.array,
                   labels: np.array, 
                   ax = None,
                   **kwargs,
                   ):
        if ax is None:
            ax = plt.gca()
        ax.barh(np.arange(len(ys)), ys, xerr=yerrs, 
               align="center",
               alpha=0.6,
               error_kw=dict(ecolor='black', elinewidth=1, capsize=4),
               **kwargs)
        ax.set_xlabel("Saliency")
        ax.set_xlim(0, ys.max()+1.3*yerrs.max())
        # ax.set_xlim(0, 0.72)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_yticks(np.arange(len(ys)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis() # invert order in yaxis
        plt.tight_layout()
        # plt.savefig("saliency_bars.png")
        # plt.show()
        return ax

    def plot_top_genes(self, 
                       topk: int = 10,
                       **kwargs):
        ys, yerrs = self.protein_j_saliency_mean[self.idx_j][:topk].detach().numpy(), self.protein_j_saliency_std[self.idx_j][:topk].detach().numpy()
        labels = self.top_genes[:topk]
        return self.plot_hbar_(ys, yerrs, labels, color=COLORS, **kwargs)

    def plot_top_TFs(self, 
                     topk: int = 10,
                     **kwargs):
        ys, yerrs = self.protein_j_saliency_mean[self.TF_intersect_idx][self.idx_j_TF][:topk].detach().numpy(), self.protein_j_saliency_std[self.TF_intersect_idx][self.idx_j_TF][:topk].detach().numpy()
        labels = self.top_TFs[:topk]
        return self.plot_hbar_(ys, yerrs, labels, color=COLORS, **kwargs)



    
