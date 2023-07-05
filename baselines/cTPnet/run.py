from baselines.mlp.data import scDataset
from torch.utils.data import Dataset, DataLoader, random_split

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray import tune, air
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from .model import cTPnetModule
from baselines.mlp.data import scDataset
from .data import savexr
from baselines.utils import *
from pathlib import Path


def _process_toy_data_var(df):
    df.index = df.index.to_series().apply(lambda x: x.split("_")[1] if "_" in x else x)


def denoise_data(X_pth,
                 X_output):
    converter = H5adToMtx(X_path=X_pth)
    converter.run(save_path=X_output, var_func=_process_toy_data_var)
    savexr(X_output, X_output)


def train(data_configs):
    train_set = scDataset(
        data_configs["X_train_pth"],
        data_configs["y_train_pth"]
    )
    n_proteins = train_set.n_proteins
    n_genes = train_set.n_genes
    seed = torch.Generator().manual_seed(42)

    train_set, valid_set = random_split(train_set,
                    [0.9, 0.1], generator=seed)  # described in the original paper
    bs = 256
    train_dl = DataLoader(train_set,
              batch_size=bs,
              shuffle=True)
    val_dl = DataLoader(valid_set,
            batch_size=bs,
            shuffle=False)
    model = cTPnetModule(n_genes=n_genes,n_proteins=n_proteins, bs=bs)

    tb_logger = TensorBoardLogger(save_dir=data_configs["result_pth"],
                  name=data_configs["name"])
    trainer = pl.Trainer(logger=tb_logger,
              max_epochs=200,
              callbacks=[
                  EarlyStopping(monitor="val_loss",
                        min_delta=0.001, patience=30,  # as described in https://github.com/zhouzilu/cTPnet/blob/master/extdata/training_05152020.py
                        verbose=False, mode="min"),
                  TuneReportCallback({"loss": "val_loss",
                                    "pearsonR": "val_pearsonR"},
                          on="validation_end")
              ],
              devices=1)
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--io", help="Path to io configs", default='io')
    parser.add_argument("-o", "--obs", help="Number of observations", default='300')
    parser.add_argument("-v", "--var", help="Number of variables", default='1000')
    parser.add_argument("-p", "--hparam", help="Path to hparam configs", default='lgb_hparams')
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
    
    if args.denoise:
        print("Running the denoise step in cTPnet")
        denoise_data(X_pth=data_configs["X_train_pth"], X_output=Path(data_configs["X_train_pth"]).parent)