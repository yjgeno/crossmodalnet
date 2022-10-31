if __name__ == "__main__":
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch 
    import json
    import sys 
    sys.path.append("/scratch/user/yjyang027/Multimodal_22/")
    from src.model import CITE_AE, MULTIOME_AE, load_model
    from src.utils import test_to_tensor, get_chrom_dicts

    MODEL_DIR = "/scratch/user/yjyang027/Multimodal_22/models/"
    DATA_DIR = "/scratch/user/yjyang027/open-problems-multimodal/"
    print(os.listdir(MODEL_DIR))

    ## cite
    MODEL_CITE = os.path.join(MODEL_DIR,"cite_multitask_L2optim")
    model_CITE = load_model(MODEL_CITE, n_input=22050, n_output=140)
    print(model_CITE)

    ada_cite = sc.read_h5ad(os.path.join(DATA_DIR, "cite_test_x.h5ad"))
    # generate cell keys
    ada_cite.obs["cell_idx"] = np.arange(len(ada_cite.obs))
    cell_id_dict_cite = ada_cite.obs["cell_idx"].to_dict() 
    with open('cell_id_dict_cite.json', 'w') as f:
        json.dump(cell_id_dict_cite, f)
    # pred
    test_x_cite = test_to_tensor(ada_cite)
    del ada_cite
    pred_y_cite = model_CITE(test_x_cite).detach().numpy()
    print("pred_y_cite:", pred_y_cite.shape)
    np.savez_compressed('pred_y_cite.npz', pred_y_cite)
    print("complete cite/n")

    ## multi
    ada_multi = sc.read_h5ad(os.path.join(DATA_DIR, "multi_test_x.h5ad"))
    ada_multi.obs["cell_idx"] = np.arange(len(ada_multi.obs))
    cell_id_dict_multi = ada_multi.obs["cell_idx"].to_dict() 
    with open('cell_id_dict_multi.json', 'w') as f:
        json.dump(cell_id_dict_multi, f)

    chrom_len_dict, chrom_idx_dict = get_chrom_dicts(ada_multi)
    print(chrom_len_dict, chrom_idx_dict)
    MODEL_MULTI = os.path.join(MODEL_DIR,"multimodal_ncorr_optim")
    model_MULTI = load_model(MODEL_MULTI, chrom_len_dict=chrom_len_dict, chrom_idx_dict=chrom_idx_dict, n_output=23418)
    print(model_MULTI)
    # pred
    test_x_multi = test_to_tensor(ada_multi)
    del ada_multi
    pred_y_multi = model_MULTI(test_x_multi).detach().numpy()
    print("pred_y_multi:", pred_y_multi.shape)
    np.savez_compressed('pred_y_multi.npz', pred_y_multi)
    print("complete multi")

