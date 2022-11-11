import torch
import numpy as np
import ray
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
import os
from .data import sc_Dataset, load_data
from .model import MULTIOME_AE, MULTIOME_DECODER
from .utils import corr_score


hyperparams = {
"lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
"seed": tune.randint(0, 10000),
"loss_ae": tune.choice(["mse", "ncorr", "comb"]),
"weight_decay": tune.sample_from(lambda _: np.random.randint(0, 9)*(0.1**np.random.randint(3, 7))),
"n_pc": tune.choice([30, 50, 100]),
"batch_size": tune.choice([128, 256, 512]),
"max_lr": tune.qloguniform(1e-1, 5e-1, 2e-2),
"step_size_up": tune.choice([10, 20, 30, 50]),
"dropout": tune.choice([0, 0.05, 0.1, 0.2]),
"batch_norm": tune.choice([True, False]),
"relu_last": tune.choice([True, False]),
"hparams_dict": {"autoencoder_width": tune.choice([[384, 1024, 384], [512]*3])} # "latent_dim": tune.choice([256, 128])
              }

def train(config): 
    torch.manual_seed(config["seed"]) 
    DIR = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "toy_data")
    # load data
    dataset = sc_Dataset(
        data_path_X = os.path.join(DIR, "multi_train_x.h5ad"), 
        data_path_Y = os.path.join(DIR, "multi_train_y.h5ad"),
        time_key = "day",
        celltype_key = "cell_type",
        preprocessing_key = "tSVD",
        prep_Y = True,
        n_components = config["n_pc"], 
        )
    train_set, val_set = load_data(dataset, batch_size = config["batch_size"])   

    # model = MULTIOME_AE(chrom_len_dict = dataset.chrom_len_dict,
    #                     chrom_idx_dict = dataset.chrom_idx_dict,
    #                     n_output= dataset.n_feature_Y,
    #                     loss_ae = config["loss_ae"],
    #                     hparams_dict = config["hparams_dict"],
    #                       )
    model = MULTIOME_DECODER(n_input = dataset.n_feature_X, 
                            n_output= dataset.n_feature_Y,
                            loss_ae = config["loss_ae"],
                            hparams_dict = config["hparams_dict"],
                            batch_norm = config["batch_norm"],
                            dropout = config["dropout"],
                            )
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = config["lr"], momentum=0.9, weight_decay = config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                  base_lr=config["max_lr"]/3, 
                                                  max_lr=config["max_lr"], 
                                                  step_size_up=config["step_size_up"], 
                                                  cycle_momentum=False,
                                                  )
    while True: # epochs < max_num_epochs
        model.train()
        loss_sum, corr_sum_train = 0., 0.
        for sample in train_set:
            X_exp, day, celltype, Y_exp = sample
            X_exp, day, celltype, Y_exp =  model.move_inputs_(X_exp, day, celltype, Y_exp)
            optimizer.zero_grad()
            components_ = torch.Tensor(dataset.processor.svd.components_)
            components_ = model.move_inputs_(components_)[0] 
            pred_Y_exp = model(X_exp, components_, relu_last=config["relu_last"])
            loss = model.loss_fn_ae(pred_Y_exp, Y_exp)
            loss_sum += loss.item()
            corr_sum_train += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())
            loss.backward()
            optimizer.step()

        # valid set
        model.eval()
        with torch.no_grad():
            corr_sum_val = 0.
            for sample in val_set:
                X_exp, day, celltype, Y_exp = sample
                X_exp, day, celltype, Y_exp =  model.move_inputs_(X_exp, day, celltype, Y_exp)
                pred_Y_exp = model(X_exp, components_, relu_last=config["relu_last"])
                corr_sum_val += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())
        
        scheduler.step()
        # record metrics from valid set
        session.report({"loss": loss_sum/len(train_set), 
                        "corr_train": corr_sum_train/len(train_set), 
                        "corr_val": corr_sum_val/len(val_set)},
                        ) # call: shown as training_iteration
    print("Finished Training")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--to_file", type=str, default="raytune_results", help="save final tuning results")
    parser.add_argument("--test", action="store_true", help="quick testing")
    args = parser.parse_args()
    
    ray.init(num_cpus = 2 if args.test else 4)
    resources_per_trial = {"cpu": 1, "gpu": 0 if args.test else 1}  # set this for GPUs
    tuner = tune.Tuner(
        tune.with_resources(train, resources = resources_per_trial),
        tune_config = tune.TuneConfig(
            metric = "corr_val",
            mode = "max",
            scheduler = ASHAScheduler(        
                max_t = 300, # max iteration
                grace_period = 10, # stop at least after this iteration
                reduction_factor = 2
                ), # for early stopping
            num_samples = 2 if args.test else args.trials, # trials
        ),
        run_config = air.RunConfig(
            name="exp",
            stop={
                "corr_val": 0.75,
                "training_iteration": 5 if args.test else 300,
            },
        ),
        param_space = hyperparams,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    save_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), f"{args.to_file}.csv")
    results.get_dataframe().to_csv(save_path)
    # python -m src.tune_multi --test

