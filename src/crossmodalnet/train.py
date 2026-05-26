import os

import torch
import torch.utils.tensorboard as tb

from .data import load_data, sc_Dataset
from .model import CrossmodalNet, load_hparams, save_hparams, save_model
from .utils import corr_score


class _NullLogger:
    """No-op logger used when no log_dir is provided."""

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass


def train(args):
    """Train CrossmodalNet given parsed CLI args."""
    dataset = sc_Dataset(
        data_path_X=args.data_x,
        data_path_Y=args.data_y,
        time_key=args.tkey,
        preprocessing_key=args.prep,
        save_prep=args.save,
    )
    train_set, val_set = load_data(dataset, batch_size=args.batch_size)

    hparams_load = None
    if args.hparams_path is not None:
        hparams_load = load_hparams(args.hparams_path)
    model = CrossmodalNet(
        n_input=dataset.n_feature_X,
        n_output=dataset.n_feature_Y,
        time_p=dataset.unique_day,
        hparams_dict=hparams_load,
    )
    print(model)
    print("hparams:", model.hparams)

    if args.optimizer == "Adam":
        opt_1 = torch.optim.Adam(model.params_ae, lr=model.hparams["ae_lr"], weight_decay=model.hparams["ae_wd"])
        opt_2 = torch.optim.Adam(model.weight_params, lr=model.hparams["weight_lr"])
        opt_adv = torch.optim.Adam(model.params_adv, lr=model.hparams["adv_lr"], weight_decay=model.hparams["adv_wd"])
    elif args.optimizer == "SGD":
        opt_1 = torch.optim.SGD(model.params_ae, lr=model.hparams["ae_lr"], momentum=0.9, weight_decay=model.hparams["ae_wd"])
        opt_2 = torch.optim.SGD(model.weight_params, lr=model.hparams["weight_lr"])
        opt_adv = torch.optim.SGD(model.params_adv, lr=model.hparams["adv_lr"], momentum=0.9, weight_decay=model.hparams["adv_wd"])
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(os.path.join("logger", args.log_dir, "train"), flush_secs=1)
    else:
        train_logger = _NullLogger()

    global_step = 0
    l0_1 = l0_2 = None
    for epoch in range(args.n_epochs):
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
                train_logger.add_scalars(
                    "loss_adv",
                    {
                        "loss_adv": loss_adv_sum / local_step_adv,
                        "adv_penalty": adv_penalty_sum / local_step_adv,
                    },
                    global_step,
                )
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
                train_logger.add_scalar("Lgrad", Lgrad.detach().item(), global_step)
                Lgrad.backward()

                opt_2.step()
                opt_1.step()

                coef = 2 / torch.add(model.weight_loss_1, model.weight_loss_2)
                model.weight_params = [coef * model.weight_loss_1, coef * model.weight_loss_2]
                local_step_ae += 1
            global_step += 1

        train_logger.add_scalars(
            "loss_ae",
            {
                "loss": loss_sum / max(local_step_ae, 1),
                "loss_1": loss_1_sum / max(local_step_ae, 1),
                "loss_2": loss_2_sum / max(local_step_ae, 1),
            },
            global_step,
        )
        train_logger.add_scalars(
            "weight_info",
            {
                "normalized_weight_loss_1": model.weight_params[0].item(),
                "normalized_weight_loss_2": model.weight_params[1].item(),
            },
            global_step,
        )
        if args.verbose:
            print(
                "(Train) epoch: {:03d}, global_step: {:d}, loss: {:.4f}, loss_adv: {:.4f}, corr: {:.4f}".format(
                    epoch,
                    global_step,
                    loss_sum / max(local_step_ae, 1),
                    loss_adv_sum / local_step_adv if local_step_adv > 0 else 0.0,
                    corr_sum_train / max(local_step_ae, 1),
                )
            )

        model.eval()
        with torch.no_grad():
            corr_sum_val = 0.0
            for X_exp, day, Y_exp in val_set:
                X_exp, day, Y_exp = model.move_inputs_(X_exp, day, Y_exp)
                pred_Y_exp = model(X_exp, day)
                corr_sum_val += corr_score(
                    Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy()
                )
            train_logger.add_scalars(
                "corr_info",
                {
                    "corr_train": corr_sum_train / max(local_step_ae, 1),
                    "corr_val": corr_sum_val / max(len(val_set), 1),
                },
                global_step,
            )
            if args.verbose:
                print(
                    "(Val) epoch: {:03d}, global_step: {:d}, corr: {:.4f}".format(
                        epoch, global_step, corr_sum_val / max(len(val_set), 1)
                    )
                )

        stop = model.early_stopping(corr_sum_val / max(len(val_set), 1))
        if stop:
            print(f"early_stop: {epoch}")
            break

    if args.save:
        save_dir = args.save_dir or "."
        save_model(model, path=save_dir)
        save_hparams(model, path=save_dir)
        print(f"Saved model and hparams to {save_dir}")

    return model
