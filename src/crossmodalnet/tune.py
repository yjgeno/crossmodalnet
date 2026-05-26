"""Hyperparameter tuning via Ray Tune.

Install with the ``tune`` extra::

    pip install "crossmodalnet[tune]"
"""

import os

import numpy as np
import torch

try:
    import ray
    from ray import train, tune
    from ray.tune.schedulers import ASHAScheduler
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Ray Tune is not installed. Install with: pip install 'crossmodalnet[tune]'"
    ) from e

from .data import load_data, sc_Dataset
from .model import CrossmodalNet
from .utils import corr_score


def build_search_space():
    return {
        "seed": tune.randint(0, 10000),
        "optimizer": tune.choice(["Adam", "SGD"]),
        "hparams_dict": {
            "first_layer_dropout": tune.choice([0, 0.05, 0.15, 0.3]),
            "adv_hidden": tune.choice([[128], [32]]),
            "reg_adv": tune.quniform(0, 2, 0.1),
            "penalty_adv": tune.quniform(0, 2, 0.1),
            "ae_lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
            "weight_lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
            "adv_lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
            "ae_wd": tune.sample_from(
                lambda _: np.random.randint(1, 10) * (0.1 ** np.random.randint(3, 7))
            ),
            "adv_wd": tune.sample_from(
                lambda _: np.random.randint(1, 10) * (0.1 ** np.random.randint(3, 7))
            ),
            "adv_step": tune.choice([3, 5, 10]),
            "alpha": tune.quniform(0, 3, 0.1),
        },
    }


def make_trainable(data_path_x: str, data_path_y: str, time_key: str = "day"):
    """Build a Ray Tune trainable bound to the given dataset paths."""

    def _train(config):
        torch.manual_seed(config["seed"])
        dataset = sc_Dataset(
            data_path_X=data_path_x,
            data_path_Y=data_path_y,
            time_key=time_key,
        )
        train_set, val_set = load_data(dataset)
        model = CrossmodalNet(
            n_input=dataset.n_feature_X,
            n_output=dataset.n_feature_Y,
            time_p=dataset.unique_day,
            hparams_dict=config["hparams_dict"],
        )

        if config["optimizer"] == "Adam":
            opt_1 = torch.optim.Adam(model.params_ae, lr=model.hparams["ae_lr"], weight_decay=model.hparams["ae_wd"])
            opt_2 = torch.optim.Adam(model.weight_params, lr=model.hparams["weight_lr"])
            opt_adv = torch.optim.Adam(model.params_adv, lr=model.hparams["adv_lr"], weight_decay=model.hparams["adv_wd"])
        else:
            opt_1 = torch.optim.SGD(model.params_ae, lr=model.hparams["ae_lr"], momentum=0.9, weight_decay=model.hparams["ae_wd"])
            opt_2 = torch.optim.SGD(model.weight_params, lr=model.hparams["weight_lr"])
            opt_adv = torch.optim.SGD(model.params_adv, lr=model.hparams["adv_lr"], momentum=0.9, weight_decay=model.hparams["adv_wd"])

        global_step = 0
        l0_1 = l0_2 = None
        while True:
            model.train()
            local_step_ae, local_step_adv = 0, 0
            loss_sum = loss_1_sum = loss_2_sum = loss_adv_sum = adv_penalty_sum = corr_sum_train = 0.0
            for X_exp, day, Y_exp in train_set:
                X_exp, day, Y_exp = model.move_inputs_(X_exp, day, Y_exp)
                pred_Y_exp, latent_base = model(X_exp, day, return_latent=True)
                adv_time_pred = model.adv_mlp(latent_base)
                adv_loss = model.loss_fn_adv(adv_time_pred, day)
                if global_step > 0 and global_step % model.hparams["adv_step"] == 0:
                    adv_penalty = model.compute_gradients(adv_time_pred.sum(), latent_base)
                    opt_adv.zero_grad()
                    loss_adv = adv_loss + model.hparams["penalty_adv"] * adv_penalty
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
                    loss = torch.div(torch.add(loss_1, loss_2), 2) - model.hparams["reg_adv"] * adv_loss
                    loss_sum += loss.item()
                    corr_sum_train += corr_score(
                        Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy()
                    )
                    opt_1.zero_grad()
                    loss.backward(retain_graph=True)

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

                    C1 = (G_avg * (inv_rate_1) ** model.hparams["alpha"]).detach().squeeze()
                    C2 = (G_avg * (inv_rate_2) ** model.hparams["alpha"]).detach().squeeze()

                    opt_2.zero_grad()
                    Lgrad = torch.add(model.grad_loss(G1, C1), model.grad_loss(G2, C2))
                    Lgrad.backward()
                    opt_2.step()
                    opt_1.step()

                    coef = 2 / torch.add(model.weight_loss_1, model.weight_loss_2)
                    model.weight_params = [coef * model.weight_loss_1, coef * model.weight_loss_2]
                    local_step_ae += 1
                global_step += 1

            model.eval()
            corr_sum_val = 0.0
            with torch.no_grad():
                for X_exp, day, Y_exp in val_set:
                    X_exp, day, Y_exp = model.move_inputs_(X_exp, day, Y_exp)
                    pred_Y_exp = model(X_exp, day)
                    corr_sum_val += corr_score(
                        Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy()
                    )

            train.report({
                "loss": loss_sum / max(local_step_ae, 1),
                "loss_1": loss_1_sum / max(local_step_ae, 1),
                "loss_2": loss_2_sum / max(local_step_ae, 1),
                "loss_adv": loss_adv_sum / local_step_adv if local_step_adv > 0 else 0.0,
                "adv_penalty": adv_penalty_sum / local_step_adv if local_step_adv > 0 else 0.0,
                "normalized_weight_loss_1": model.weight_params[0].item(),
                "normalized_weight_loss_2": model.weight_params[1].item(),
                "corr_train": corr_sum_train / max(local_step_ae, 1),
                "corr_val": corr_sum_val / max(len(val_set), 1),
            })

    return _train


def run_tune(
    data_path_x: str,
    data_path_y: str,
    time_key: str = "day",
    trials: int = 50,
    test: bool = False,
    max_t: int = 300,
    grace_period: int = 10,
    output_csv: str = "raytune_results.csv",
):
    """Run Ray Tune hyperparameter search."""
    ray.init(num_cpus=2 if test else max(os.cpu_count() - 1, 1))
    resources_per_trial = {"cpu": 1, "gpu": 0 if test else 1}
    tuner = tune.Tuner(
        tune.with_resources(
            make_trainable(data_path_x, data_path_y, time_key),
            resources=resources_per_trial,
        ),
        tune_config=tune.TuneConfig(
            metric="corr_val",
            mode="max",
            scheduler=ASHAScheduler(max_t=max_t, grace_period=grace_period, reduction_factor=2),
            num_samples=2 if test else trials,
        ),
        run_config=train.RunConfig(
            name="exp",
            stop={"corr_val": 0.95, "training_iteration": 5 if test else max_t},
        ),
        param_space=build_search_space(),
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    results.get_dataframe().to_csv(output_csv)
    return results
