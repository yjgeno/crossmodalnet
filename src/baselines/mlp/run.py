import argparse

from time import time
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray import tune, air
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from .model import Module
from .data import scDataset
from src.baselines.utils import *
from pathlib import Path


def train(hparams, data_configs, use_ncorr_loss=False):
    train_set = scDataset(
        data_configs["X_train_pth"],
        data_configs["y_train_pth"]
              )
    n_proteins = train_set.n_proteins
    n_genes = train_set.n_genes
    seed = torch.Generator().manual_seed(42)

    train_set, valid_set = random_split(train_set,
                    [0.8, 0.2], generator=seed)
    bs = hparams.pop("bs", 512)
    train_dl = DataLoader(train_set,
              batch_size=bs,
              shuffle=True)
    val_dl = DataLoader(valid_set,
            batch_size=bs,
            shuffle=False)
    hparams["optim_params"] = {"lr": hparams.pop("lr", 1e-3),
                 "optim": hparams.pop("optim", "AdamW"),
                 "weight_decay": hparams.pop("weight_decay", 1e-2)}
    model = Module(n_genes=n_genes,n_proteins=n_proteins, bs=bs, use_ncorr_loss=use_ncorr_loss, **hparams)

    tb_logger = TensorBoardLogger(save_dir=data_configs["result_pth"],
                  name=data_configs["name"])
    trainer = pl.Trainer(logger=tb_logger,
              max_epochs=50,
              callbacks=[
                  EarlyStopping(monitor="val_pearsonR",
                        min_delta=0.00, patience=3,
                        verbose=False, mode="max"),
                  TuneReportCallback({"loss": "val_loss",
                            "pearsonR": "val_pearsonR"},
                          on="validation_end")
              ],
              devices=1)
    trainer.fit(model, train_dl, val_dl)


def test(hparams,
         train_set,
         test_set,
         data_configs,
         use_ncorr_loss=False):
    bs = hparams.pop("bs", 512)
    n_proteins = train_set.n_proteins
    n_genes = train_set.n_genes
    train_dl = DataLoader(train_set,
                          batch_size=bs,
                          shuffle=True)
    test_dl = DataLoader(test_set,
                         batch_size=bs,
                         shuffle=False)
    hparams["optim_params"] = {"lr": hparams.pop("lr", 1e-3),
                               "optim": hparams.pop("optim", "AdamW"),
                               "weight_decay": hparams.pop("weight_decay", 1e-2)}
    model = Module(n_genes=n_genes, n_proteins=n_proteins, use_ncorr_loss=use_ncorr_loss, **hparams)

    tb_logger = TensorBoardLogger(save_dir=data_configs["result_pth"],
                                  name=data_configs["name"])
    trainer = pl.Trainer(logger=tb_logger,
                         max_epochs=30,
                         callbacks=[],
                         devices=1,
                         limit_val_batches=0)
    start_time = time()
    trainer.fit(model=model, train_dataloaders=train_dl)
    trainer.test(ckpt_path="best", dataloaders=test_dl)
    return time() - start_time


def train_eval_time(hparams,
                    train_set,
                    use_ncorr_loss=False):
    bs = hparams.pop("bs", 512)
    n_proteins = train_set.n_proteins
    n_genes = train_set.n_genes
    train_dl = DataLoader(train_set,
                          batch_size=bs,
                          shuffle=True)
    hparams["optim_params"] = {"lr": hparams.pop("lr", 1e-3),
                               "optim": hparams.pop("optim", "AdamW"),
                               "weight_decay": hparams.pop("weight_decay", 1e-2)}
    model = Module(n_genes=n_genes, n_proteins=n_proteins, use_ncorr_loss=use_ncorr_loss, **hparams)
    trainer = pl.Trainer(max_epochs=30,
                         callbacks=[],
                         devices=1,
                         limit_val_batches=0)
    start_time = time()
    trainer.fit(model=model, train_dataloaders=train_dl)
    return time() - start_time


def tune_model(hparams, data_configs, use_ncorr_loss=False):
    scheduler = ASHAScheduler()
    reporter = CLIReporter(metric_columns=["training_iteration", "loss", "pearsonR"])
    train_func_with_param = tune.with_parameters(train,
                                                 data_configs=data_configs,
                                                 use_ncorr_loss=use_ncorr_loss)
    resources = {"cpu": 1}
    tuner = tune.Tuner(
      tune.with_resources(train_func_with_param,
                      resources=resources),
      tune_config=tune.TuneConfig(
          metric="loss",
          mode="min",
          scheduler=scheduler,
          num_samples=20,
      ),
      run_config=air.RunConfig(local_dir=data_configs["result_pth"] + "/result",
                   name=f"experiment_{data_configs['name']}_{'mse' if not use_ncorr_loss else 'ncorr'}",
                   log_to_file=("stdout.log", "stderr.log"),
                   progress_reporter=reporter),
      param_space=hparams
    )
    results = tuner.fit()
    print(results.get_best_result(metric="pearsonR", mode='max').config)
    result_df = results.get_dataframe(filter_metric="pearsonR", filter_mode="max")
    result_df.to_csv(data_configs["result_pth"] + f"/result/{data_configs['name']}_{'mse' if not use_ncorr_loss else 'ncorr'}_tune_result.csv")
    train_set = scDataset(data_configs["X_train_pth"],
                          data_configs["y_train_pth"])
    test_set = scDataset(data_configs["X_test_pth"],
                         data_configs["y_test_pth"])
    test(results.get_best_result(metric="pearsonR", mode='max').config,
         train_set=train_set,
         test_set=test_set,
         data_configs=data_configs,
         use_ncorr_loss=use_ncorr_loss)


if __name__ == "__main__":
    hparams = {"activation": tune.choice(["SELU", "LeakyReLU", "Hardswish", "Sigmoid"]),
               "dropout_rate": tune.uniform(0.01, 0.5),
               # "loss": tune.choice(["huber_loss", "mse_loss"]),  <- use ncorr
               "hidden_dims": tune.choice([[512, 216],
                              [1024, 512, 216],
                              [512, 512, 512],
                              [1024, 216, 64]]),
               "lr": tune.loguniform(1e-4, 1e-1),
               "optim": tune.choice(["AdamW", "Adam", "SGD"]),
               "weight_decay": tune.loguniform(1e-4, 8e-1),
               "bs": tune.choice([512, 1024, 2048]),
               "use_layernorm": tune.choice([False, True]),
               "use_batchnorm": tune.choice([False, True]),}

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--ncorr", help="Use NCorr loss", action='store_true')
    parser.add_argument("-i", "--io", help="Path to io configs", default='io')
    parser.add_argument("-o", "--obs", help="Number of observations", default='300')
    parser.add_argument("-v", "--var", help="Number of variables", default='1000')
    parser.add_argument("-p", "--hparam", help="Path to hparam configs", default='mlp_hparams')
    parser.add_argument("-e", "--eval_time", help="To evaluate training time or not",
                        action='store_true')
    args = parser.parse_args()
    if args.ncorr:
        hparams.update({"loss": tune.choice(["huber_loss", "mse_loss"])})

    cfd = Path(__file__).resolve().parent.parent / "configs"
    data_configs = load_config((cfd / args.io).with_suffix(".yml")) \
        if (cfd / args.io).with_suffix(".yml").is_file() else load_config(args.io)

    if args.eval_time:
        X_train, y_train = sc.read_h5ad(data_configs["X_train_pth"]), sc.read_h5ad(data_configs["y_train_pth"])
        X_train, y_train = get_subset(X_train, y_train,
                                      n_obs=int(args.obs),
                                      n_vars=int(args.var))
        train_ds = scDataset.init_with_data(X_train, y_train)
        best_hparams = load_config((cfd / args.hparam).with_suffix(".yml")) \
            if (cfd / args.hparam).with_suffix(".yml").is_file() else load_config(args.hparam)
        est_time = train_eval_time(best_hparams, train_ds, use_ncorr_loss=args.ncorr)
        result_dir = Path(data_configs["result_dir_pth"]).mkdir(parents=True, exist_ok=True)
        print_info(est_time, n_obs=int(args.obs), n_var=int(args.var), hparams=best_hparams,
                   file_name= Path(data_configs["result_dir_pth"]) / f"mlp.csv")
    else:
        tune_model(hparams, data_configs, use_ncorr_loss=args.ncorr)
