import torch
import numpy as np
import ray
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
import os
from .data import sc_Dataset, load_data
from .model import CITE_AE
from .utils import corr_score


hyperparams = {
"lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
"seed": tune.randint(0, 10000),
"optimizer": tune.choice(["Adam", "SGD"]),
"loss_ae": tune.choice(["mse", "ncorr", "gauss", "nb",]),
"weight_decay": tune.sample_from(lambda _: np.random.randint(1, 10)*(0.1**np.random.randint(3, 7))),
"hparams_dict": {"latent_dim": tune.choice([64, 32]), 
                 "autoencoder_width": tune.choice([[512, 128], [256]])},
                             }

def train(config): 
    torch.manual_seed(config["seed"]) 
    DIR = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "toy_data")
    # load data
    dataset = sc_Dataset(
            data_path_X = os.path.join(DIR, "cite_train_x.h5ad"), # HARD coded only for tune
            data_path_Y = os.path.join(DIR, "cite_train_y.h5ad"),
            time_key = "day",
            celltype_key = "cell_type",
            )
    train_set, val_set = load_data(dataset)   
    model = CITE_AE(n_input = dataset.n_feature_X, 
                    n_output= dataset.n_feature_Y,
                    loss_ae = config["loss_ae"],
                    hparams_dict = config["hparams_dict"],
                    )
    # optimizer
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = config["lr"], momentum = 0.9, weight_decay = config["weight_decay"])

    while True: # epochs < max_num_epochs
        model.train()
        loss_sum, corr_sum_train = 0., 0.
        for sample in train_set:
            X_exp, day, celltype, Y_exp = sample
            X_exp, day, celltype, Y_exp =  model.move_inputs_(X_exp, day, celltype, Y_exp)
            optimizer.zero_grad()
            pred_Y_exp = model(X_exp)
            loss = model.loss_fn_ae(pred_Y_exp, Y_exp)
            loss_sum += loss.item()
            if model.loss_ae in model.loss_type2:
                    pred_Y_exp = model.sample_pred_from(pred_Y_exp)
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
                pred_Y_exp = model(X_exp)
                if model.loss_ae in model.loss_type2:
                    pred_Y_exp = model.sample_pred_from(pred_Y_exp)
                corr_sum_val += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())

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
                max_t = 50, # max iteration
                grace_period = 10, # stop at least after this iteration
                reduction_factor = 2
                ), # for early stopping
            num_samples = 2 if args.test else args.trials, # trials
        ),
        run_config = air.RunConfig(
            name="exp",
            stop={
                "corr_val": 0.98,
                "training_iteration": 5 if args.test else 50,
            },
        ),
        param_space = hyperparams,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    save_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), f"{args.to_file}.csv")
    results.get_dataframe().to_csv(save_path)
    # python -m src.tune_cite --test

