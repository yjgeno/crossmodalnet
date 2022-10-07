import torch
import ray
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
import os
from .data import sc_Dataset, load_data
from .model import multimodal_AE
from .utils import corr_score


hyperparams = {
"lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
"seed": tune.randint(0, 10000),
"optimizer": tune.choice(["Adam", "SGD"]),
"loss_ae": tune.choice(["mse", "ncorr", "gauss", "nb", "custom_"]),
"alpha": tune.quniform(0, 1, 0.1),
"beta": tune.quniform(0, 1, 0.1),
}


def train(config): 
    DIR = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "toy_data")
    # load data
    dataset = sc_Dataset(
            data_path_X = os.path.join(DIR, "cite_train_x.h5ad"), # HARD coded only for tune
            data_path_Y = os.path.join(DIR, "cite_train_y.h5ad"),
            time_key = "day",
            celltype_key = "cell_type",
            )
    train_set, val_set = load_data(dataset)
    torch.manual_seed(config["seed"]) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = multimodal_AE(n_input = dataset.n_feature_X, 
                          n_output= dataset.n_feature_Y,
                          loss_ae = config["loss_ae"],
                          )
    model = model.to(device)  
    # optimizer
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], weight_decay = 1e-5)
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = config["lr"], momentum = 0.9, weight_decay = 5e-4)

    while True: # epochs < max_num_epochs
        model.train()
        loss_sum, corr_sum_train = 0., 0.
        for sample in train_set:
            X_exp, day, celltype, Y_exp = sample
            X_exp, day, celltype, Y_exp =  X_exp.to(device), day.to(device), celltype.to(device), Y_exp.to(device)
            optimizer.zero_grad()
            pred_Y_exp = model(X_exp)
            if model.loss_ae == "custom_":
                alpha, beta = config["alpha"], config["beta"] 
                pred_Y_means = pred_Y_exp[:, :model.n_output]
                loss = model.loss_fn_mse(pred_Y_means, Y_exp) + alpha*model.loss_fn_ncorr(pred_Y_means, Y_exp) + beta*model.loss_fn_gauss(pred_Y_exp, Y_exp)
            else:
                loss = model.loss_fn_ae(pred_Y_exp, Y_exp)
            loss_sum += loss.item()
            if model.loss_ae in model.loss_type2:
                    pred_Y_exp = model.sample_pred_from(pred_Y_exp)
            corr_sum_train += corr_score(Y_exp.detach().numpy(), pred_Y_exp.detach().numpy())
            loss.backward()
            optimizer.step()

        # valid set
        model.eval()
        with torch.no_grad():
            corr_sum_val = 0.
            for sample in val_set:
                X_exp, day, celltype, Y_exp = sample
                X_exp, day, celltype, Y_exp =  X_exp.to(device), day.to(device), celltype.to(device), Y_exp.to(device)
                pred_Y_exp = model(X_exp)
                if model.loss_ae in model.loss_type2:
                    pred_Y_exp = model.sample_pred_from(pred_Y_exp)
                corr_sum_val += corr_score(Y_exp.detach().numpy(), pred_Y_exp.detach().numpy())

        # record metrics from valid set
        session.report({"loss": loss_sum/len(train_set), 
                        "corr_train": corr_sum_train/len(train_set), 
                        "corr_val": corr_sum_val/len(val_set)},
                        ) # call: shown as training_iteration
    print("Finished Training")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="quick testing")
    args = parser.parse_args()
    
    ray.init(num_cpus = 2 if args.test else None)
    resources_per_trial = {"cpu": 1, "gpu": 0}  # set this for GPUs
    tuner = tune.Tuner(
        tune.with_resources(train, resources = resources_per_trial),
        tune_config = tune.TuneConfig(
            metric = "corr_val",
            mode = "max",
            scheduler = ASHAScheduler(        
                max_t = 100, # max iteration
                grace_period = 10, # stop at least after this iteration
                reduction_factor = 2
                ), # for early stopping
            num_samples = 2 if args.test else 50, # trials
        ),
        run_config = air.RunConfig(
            name="exp",
            stop={
                "corr_val": 0.98,
                "training_iteration": 5 if args.test else 100,
            },
        ),
        param_space = hyperparams,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    # python -m src.tune --test

