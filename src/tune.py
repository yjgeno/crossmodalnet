import torch
import numpy as np
import ray
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
import os
from .data import sc_Dataset, load_data
from .model import multimodal_AE
from .utils import corr_score


hyperparams = {
"lr_1": tune.qloguniform(1e-4, 1e-1, 5e-5),
"lr_2": tune.qloguniform(1e-4, 1e-1, 5e-5),
"seed": tune.randint(0, 10000),
"optimizer": tune.choice(["Adam", "SGD"]),
"weight_decay": tune.sample_from(lambda _: np.random.randint(1, 10)*(0.1**np.random.randint(3, 7))),
"hparams_dict": {"latent_dim": tune.choice([64, 32]), 
                 "autoencoder_width": tune.choice([[512, 128], [256]]),
                 "alpha": tune.quniform(0, 3, 0.1)
                }
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
    model = multimodal_AE(n_input = dataset.n_feature_X, 
                          n_output= dataset.n_feature_Y,
                          loss_ae = "multitask",
                          hparams_dict = config["hparams_dict"], # set alpha
                          )
    
    # optimizer
    if config["optimizer"] == "Adam":
        opt_1 = torch.optim.Adam(model.parameters(), lr = config["lr_1"], weight_decay = config["weight_decay"])
        opt_2 = torch.optim.Adam(model.weight_params, lr = config["lr_2"])
    elif config["optimizer"] == "SGD":
        opt_1 = torch.optim.SGD(model.parameters(), lr = config["lr_1"], momentum = 0.9, weight_decay = config["weight_decay"])
        opt_2 = torch.optim.SGD(model.weight_params, lr = config["lr_2"])

    FIRST_STEP = True
    while True: # epochs < max_num_epochs
        model.train()
        loss_sum, loss_1_sum, loss_2_sum, corr_sum_train = 0., 0., 0., 0.
        for sample in train_set:
            X_exp, day, celltype, Y_exp = sample
            X_exp, day, celltype, Y_exp = model.move_inputs_(X_exp, day, celltype, Y_exp)
            
            pred_Y_exp = model(X_exp)
            loss_1 = model.weight_params[0] * model.loss_fn_1(pred_Y_exp, Y_exp)
            loss_2 = model.weight_params[1] * model.loss_fn_2(pred_Y_exp, Y_exp)
            if FIRST_STEP:
                l0_1, l0_2 = loss_1.detach(), loss_2.detach()
                FIRST_STEP = False
            loss_1_sum += loss_1.item()
            loss_2_sum += loss_2.item()
            loss = torch.div(torch.add(loss_1,loss_2), 2)
            loss_sum += loss.item()
            corr_sum_train += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())
            opt_1.zero_grad()
            loss.backward(retain_graph=True) # retain_graph for G1R and G2R
            
            # calculate norm of current gradient w.r.t. 1st layer param W
            W = list(model.parameters())[0] 
            G1R = torch.autograd.grad(loss_1, W, retain_graph=True, create_graph=True) 
            G1 = torch.norm(G1R[0], 2) 
            G2R = torch.autograd.grad(loss_2, W, retain_graph=True, create_graph=True)
            G2 = torch.norm(G2R[0], 2)
            G_avg = torch.div(torch.add(G1, G2), 2) 
            
            # calculate relative losses 
            lhat_1 = torch.div(loss_1, l0_1)
            lhat_2 = torch.div(loss_2, l0_2)
            lhat_avg = torch.div(torch.add(lhat_1, lhat_2), 2)
            
            # calculate relative inverse training rates: lower -> train faster
            inv_rate_1 = torch.div(lhat_1, lhat_avg)
            inv_rate_2 = torch.div(lhat_2, lhat_avg)
            
            # calculate the constant target 
            C1 = G_avg*(inv_rate_1)**model.hparams["alpha"]
            C2 = G_avg*(inv_rate_2)**model.hparams["alpha"]
            C1 = C1.detach().squeeze() 
            C2 = C2.detach().squeeze()
            
            opt_2.zero_grad()
            # calculate the gradient loss
            Lgrad = torch.add(model.grad_loss(G1, C1), model.grad_loss(G2, C2))
            Lgrad.backward()        
            # update
            opt_2.step()
            opt_1.step()
            
            # normalize the loss weights: sum to 2
            coef = 2/torch.add(model.weight_loss_1, model.weight_loss_2)
            model.weight_params = [coef*model.weight_loss_1, coef*model.weight_loss_2] # update

        model.eval()
        with torch.no_grad():
            corr_sum_val = 0.
            for sample in val_set:
                X_exp, day, celltype, Y_exp = sample
                X_exp, day, celltype, Y_exp = model.move_inputs_(X_exp, day, celltype, Y_exp)
                pred_Y_exp = model(X_exp)
                corr_sum_val += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())     

        # record metrics
        session.report({"loss": loss_sum/len(train_set), 
                        "loss_1": loss_1_sum/len(train_set),
                        "loss_2": loss_2_sum/len(train_set),
                        "normalized_weight_loss_1": model.weight_params[0].item(), 
                        "normalized_weight_loss_2": model.weight_params[1].item(), 
                        "corr_train": corr_sum_train/len(train_set),
                        "corr_val": corr_sum_val/len(val_set)},
                        ) # call: shown as training_iteration


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--to_file", type=str, default="raytune_results", help="save final tuning results")
    parser.add_argument("--test", action="store_true", help="quick testing")
    args = parser.parse_args()
    
    ray.init(num_cpus = 2 if args.test else os.cpu_count()-1)
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
    # python -m src.tune --test

