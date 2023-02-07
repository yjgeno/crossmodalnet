import torch
import numpy as np
import ray
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
import os
from .data import sc_Dataset, load_data
from .model import CrossmodalNet
from .utils import corr_score


hyperparams = {
"seed": tune.randint(0, 10000),
"optimizer": tune.choice(["Adam", "SGD"]),
# "batch_norm": tune.choice([True, False]),
"hparams_dict": {
                # "n_latent": tune.choice([512, 256]),
                "first_layer_dropout": tune.choice([0, 0.05, 0.15, 0.3]),
                # "encoder_hidden": [512, 512], 
                "adv_hidden": tune.choice([[128], [32]]),
                "reg_adv": tune.quniform(0, 2, 0.1),
                "penalty_adv": tune.quniform(0, 2, 0.1),
                "ae_lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
                "weight_lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
                "adv_lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
                "ae_wd": tune.sample_from(lambda _: np.random.randint(1, 10)*(0.1**np.random.randint(3, 7))),
                "adv_wd": tune.sample_from(lambda _: np.random.randint(1, 10)*(0.1**np.random.randint(3, 7))),
                "adv_step": tune.choice([3, 5]),
                "alpha": tune.quniform(0, 3, 0.1),
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
            # celltype_key = "cell_type",
            )
    train_set, val_set = load_data(dataset)  
    model = CrossmodalNet(n_input = dataset.n_feature_X, 
                          n_output= dataset.n_feature_Y,
                          time_p = dataset.unique_day,
                          hparams_dict = config["hparams_dict"],
                          )

    if config["optimizer"] == "Adam":
        opt_1 = torch.optim.Adam(model.params_ae, lr=model.hparams["ae_lr"], weight_decay=model.hparams["ae_wd"])
        opt_2 = torch.optim.Adam(model.weight_params, lr=model.hparams["weight_lr"]) # no decay for loss weight
        opt_adv = torch.optim.Adam(model.params_adv, lr=model.hparams["adv_lr"], weight_decay=model.hparams["adv_wd"])
    elif config["optimizer"] == "SGD":
        opt_1 = torch.optim.SGD(model.params_ae, lr=model.hparams["ae_lr"], momentum=0.9, weight_decay=model.hparams["ae_wd"])
        opt_2 = torch.optim.SGD(model.weight_params, lr=model.hparams["weight_lr"])
        opt_adv = torch.optim.SGD(model.params_adv, lr= model.hparams["adv_lr"], momentum=0.9, weight_decay=model.hparams["adv_wd"])

    global_step = 0
    while True: # epochs < max_num_epochs
        model.train()    
        local_step_ae, local_step_adv = 0, 0
        loss_sum, loss_1_sum, loss_2_sum, loss_adv_sum, adv_penalty_sum, corr_sum_train = [0.]*6
        for sample in train_set:        
            X_exp, day, Y_exp = sample
            X_exp, day, Y_exp = model.move_inputs_(X_exp, day, Y_exp)   
            pred_Y_exp, latent_base = model(X_exp, day, return_latent=True)
            # adv pred
            adv_time_pred = model.adv_mlp(latent_base)
            adv_loss = model.loss_fn_adv(adv_time_pred, day)
            if global_step > 0 and global_step%model.hparams["adv_step"]==0:
                adv_penalty = model.compute_gradients(adv_time_pred.sum(), latent_base)
                opt_adv.zero_grad()
                loss_adv = adv_loss + model.hparams["penalty_adv"]*adv_penalty
                loss_adv_sum += loss_adv.item()
                adv_penalty_sum += adv_penalty.item()
                loss_adv.backward()
                opt_adv.step()
                local_step_adv += 1
            
            else:
                loss_1 = model.weight_params[0] * model.loss_fn_1(pred_Y_exp, Y_exp) 
                loss_2 = model.weight_params[1] * model.loss_fn_2(pred_Y_exp, Y_exp)
                if global_step == 0:
                    l0_1, l0_2 = loss_1.detach(), loss_2.detach()
                loss_1_sum += loss_1.item()
                loss_2_sum += loss_2.item()
                loss = torch.div(torch.add(loss_1,loss_2), 2) - model.hparams["reg_adv"]*adv_loss
                loss_sum += loss.item()
                corr_sum_train += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())
                opt_1.zero_grad()
                loss.backward(retain_graph=True) # retain_graph for G1R and G2R
                    
                W = list(model.parameters())[0] 
                G1R = torch.autograd.grad(loss_1, W, retain_graph=True, create_graph=True) 
                G1 = torch.norm(G1R[0], 2) 
                G2R = torch.autograd.grad(loss_2, W, retain_graph=True, create_graph=True) 
                G2 = torch.norm(G2R[0], 2)
                G_avg = torch.div(torch.add(G1, G2), 2) 
                
                lhat_1 = torch.div(loss_1, l0_1)
                lhat_2 = torch.div(loss_2, l0_2)
                lhat_avg = torch.div(torch.add(lhat_1, lhat_2), 2)       
                inv_rate_1 = torch.div(lhat_1, lhat_avg)
                inv_rate_2 = torch.div(lhat_2, lhat_avg)
                
                C1 = G_avg*(inv_rate_1)**model.hparams["alpha"]
                C2 = G_avg*(inv_rate_2)**model.hparams["alpha"]
                C1 = C1.detach().squeeze() 
                C2 = C2.detach().squeeze()

                opt_2.zero_grad() 
                Lgrad = torch.add(model.grad_loss(G1, C1), model.grad_loss(G2, C2)) 
                Lgrad.backward()
                opt_2.step()
                opt_1.step()
            
                coef = 2/torch.add(model.weight_loss_1, model.weight_loss_2)
                model.weight_params = [coef*model.weight_loss_1, coef*model.weight_loss_2] 
                local_step_ae += 1
            global_step += 1

        model.eval()
        with torch.no_grad():
            corr_sum_val = 0.
            for sample in val_set:
                X_exp, day, Y_exp = sample
                X_exp, day, Y_exp = model.move_inputs_(X_exp, day, Y_exp)
                pred_Y_exp = model(X_exp, day)
                corr_sum_val += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())     

        # record metrics
        session.report({"loss": loss_sum/local_step_ae,
                        "loss_1": loss_1_sum/local_step_ae,
                        "loss_2": loss_2_sum/local_step_ae,
                        "loss_adv": loss_adv_sum/local_step_adv if local_step_adv>0 else 0., 
                        "adv_penalty": adv_penalty_sum/local_step_adv if local_step_adv>0 else 0.,
                        "normalized_weight_loss_1": model.weight_params[0].item(), 
                        "normalized_weight_loss_2": model.weight_params[1].item(), 
                        "corr_train": corr_sum_train/local_step_ae,
                        "corr_val": corr_sum_val/len(val_set)},
                        ) # call: shown as training_iteration


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
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
                max_t = 300, # max iteration
                grace_period = 10, # stop at least after this iteration
                reduction_factor = 2
                ), # for early stopping
            num_samples = 2 if args.test else args.trials, # trials
        ),
        run_config = air.RunConfig(
            name="exp",
            stop={
                "corr_val": 0.95,
                "training_iteration": 5 if args.test else 300,
            },
        ),
        param_space = hyperparams,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    # save_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    results.get_dataframe().to_csv("raytune_results.csv")
    # python -m src.tune --test

