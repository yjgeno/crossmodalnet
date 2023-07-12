from baselines.mlp.data import scDataset
from torch.utils.data import Dataset, DataLoader, random_split

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from .model import cTPnetModule
from .data import cTPnetDataset
from .data import savexr
from baselines.utils import *
from pathlib import Path
from time import time


def _process_toy_data_var(df):
    df.index = df.index.to_series().apply(lambda x: x.split("_")[1] if "_" in x else x)


def denoise_data(X_pth,
                 X_output,
                 pretrained):
    converter = H5adToMtx(X_path=X_pth)
    converter.run(save_path=X_output, var_func=None)
    savexr(X_output, X_output, pretrained)


def train(data_configs, max_epochs=200):
    train_set = cTPnetDataset(
        X_path=data_configs["X_train_processed_dir"],
        y_path=data_configs["y_train_pth"]
    )
    n_proteins = train_set.n_proteins
    n_genes = train_set.n_genes
    seed = torch.Generator().manual_seed(42)

    train_set, valid_set = random_split(train_set,
                    [0.9, 0.1], generator=seed)  # described in the original paper
    bs = 1024
    train_dl = DataLoader(train_set,
                        batch_size=bs,
                        shuffle=True)
    val_dl = DataLoader(valid_set,
                        batch_size=bs,
                        shuffle=False)
    model = cTPnetModule(n_genes=n_genes,n_proteins=n_proteins)

    tb_logger = TensorBoardLogger(save_dir=data_configs["result_dir_pth"],
                                  name=data_configs["name"])
    trainer = pl.Trainer(logger=tb_logger,
              max_epochs=max_epochs,
              callbacks=[
                  EarlyStopping(monitor="val_loss",
                        min_delta=0.001, patience=30,  # as described in https://github.com/zhouzilu/cTPnet/blob/master/extdata/training_05152020.py
                        verbose=False, mode="min"),
              ],
              devices=1)
    trainer.fit(model, train_dl, val_dl)
    return trainer


def test(data_configs):
    trainer = train(data_configs=data_configs)
    test_set = cTPnetDataset(
        X_path=data_configs["X_test_processed_dir"],
        y_path=data_configs["y_test_pth"]
    )
    bs = 1024
    test_dl = DataLoader(test_set,
                        batch_size=bs,
                        shuffle=False)
    result = trainer.test(ckpt_path="best", dataloaders=test_dl)
    pd.DataFrame(result).to_csv(Path(data_configs["result_dir_pth"]) / "test_result.csv")


def create_dummy_data(X_train, y_train):
    dummy_path = {"parent": "./tmp",
                  "X_train_pth": "./tmp/X_train.h5ad",
                  "X_train_processed_dir": "./tmp/p_train_x",
                  "y_train_pth": "./tmp/y_train.h5ad",}
    Path(dummy_path["parent"]).mkdir(exist_ok=True)
    Path(dummy_path["X_train_processed_dir"]).mkdir(exist_ok=True)
    X_train.write_h5ad(dummy_path["X_train_pth"])
    y_train.write_h5ad(dummy_path["y_train_pth"])
    return dummy_path


def train_eval_time(configs):
    print("Running the denoise step in cTPnet")
    start_time = time()
    denoise_data(X_pth=configs["X_train_pth"], 
                    X_output=configs["X_train_processed_dir"],
                    pretrained=configs["pretrained_weights"])
    train(data_configs=configs, max_epochs=2)
    return time() - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--io", help="Path to io configs", default='io')
    parser.add_argument("-o", "--obs", help="Number of observations", default='300')
    parser.add_argument("-v", "--var", help="Number of variables", default='1000')
    parser.add_argument("-e", "--eval_time", help="To evaluate training time or not",
                        action='store_true')
    parser.add_argument("-d", "--denoise", help="Whether to run denoise step",
                        action='store_true')
    parser.add_argument("-m", "--eval_memory", help="To evaluate memory usage or not",
                        action='store_true')

    args = parser.parse_args()

    cfd = Path(__file__).resolve().parent.parent / "configs"
    data_configs = load_config((cfd / args.io).with_suffix(".yml")) \
        if (cfd / args.io).with_suffix(".yml").is_file() else load_config(args.io)
    if args.eval_time:
        X_train, y_train = sc.read_h5ad(data_configs["X_train_pth"]), sc.read_h5ad(data_configs["y_train_pth"])
        X_train, y_train = get_subset(X_train, y_train,
                                      n_obs=int(args.obs),
                                      n_vars=int(args.var), replace=False)
        dummy_configs = create_dummy_data(X_train, y_train)
        data_configs.update(dummy_configs)
        
        if args.eval_memory:
            with open(Path(data_configs["result_dir_pth"]) / data_configs["mem_file_name"], "w+") as fp:
                est_time = train_eval_time_mem(train_eval_time, fp)(data_configs)
            est_mem = extract_peak_mem(Path(data_configs["result_dir_pth"]) / data_configs["mem_file_name"])
        else:
            est_time = train_eval_time(data_configs)
            est_mem = None
        result_dir = Path(data_configs["result_dir_pth"]).mkdir(parents=True, exist_ok=True)
        print_info(est_time, n_obs=int(args.obs), n_var=int(args.var), hparams={}, memory_used=est_mem,
                   file_name= Path(data_configs["result_dir_pth"]) / f"ctpnet.csv")

    else:
        if args.denoise:
            print("Running the denoise step in cTPnet")
            denoise_data(X_pth=data_configs["X_train_pth"], 
                        X_output=data_configs["X_train_processed_dir"],
                        pretrained=data_configs["pretrained_weights"])
            denoise_data(X_pth=data_configs["X_test_pth"], 
                        X_output=data_configs["X_test_processed_dir"],
                        pretrained=data_configs["pretrained_weights"])
        
        print("Running model training and testing part")
        test(data_configs=data_configs)
