import torch
import os
import torch.utils.tensorboard as tb
from .data import sc_Dataset, load_data
from .model import CrossmodalNet, save_model, save_hparams, load_hparams
from .utils import corr_score


def train(args):
    """
    Train model.
    """
    # load data
    dataset = sc_Dataset(
            data_path_X = os.path.join(args.data_dir, "cite_train_x.h5ad"),
            data_path_Y = os.path.join(args.data_dir, "cite_train_y.h5ad"),
            time_key = "day",
            # celltype_key = "cell_type",
            preprocessing_key = args.prep,
            save_prep = args.save,
            )
    train_set, val_set = load_data(dataset, batch_size = args.batch_size)

    # init model
    hparams_load = None
    if args.hparams_path is not None:
        hparams_load = load_hparams(os.path.join("hparams", args.hparams_path))
    model = CrossmodalNet(n_input = dataset.n_feature_X, 
                          n_output= dataset.n_feature_Y,
                          time_p = dataset.unique_day,
                          hparams_dict = hparams_load,
                          )
    print(model)
    print("hparams:", model.hparams)

    # optimizer
    if args.optimizer == "Adam":
        opt_1 = torch.optim.Adam(model.params_ae, lr=model.hparams["ae_lr"], weight_decay=model.hparams["ae_wd"])
        opt_2 = torch.optim.Adam(model.weight_params, lr=model.hparams["weight_lr"]) # no decay for loss weight
        opt_adv = torch.optim.Adam(model.params_adv, lr=model.hparams["adv_lr"], weight_decay=model.hparams["adv_wd"])
    elif args.optimizer == "SGD":
        opt_1 = torch.optim.SGD(model.params_ae, lr=model.hparams["ae_lr"], momentum=0.9, weight_decay=model.hparams["ae_wd"])
        opt_2 = torch.optim.SGD(model.weight_params, lr=model.hparams["weight_lr"])
        opt_adv = torch.optim.SGD(model.params_adv, lr= model.hparams["adv_lr"], momentum=0.9, weight_decay=model.hparams["adv_wd"])

    
    # logging
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(os.path.join("logger", args.log_dir, "train"), flush_secs=1)
        # valid_logger = tb.SummaryWriter(os.path.join("logger", args.log_dir, 'valid'), flush_secs=1)
    global_step = 0

    for epoch in range(args.n_epochs):
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
                # print("step_adv", global_step, local_step_adv)
                adv_penalty = model.compute_gradients(adv_time_pred.sum(), latent_base)
                opt_adv.zero_grad()
                loss_adv = adv_loss + model.hparams["penalty_adv"]*adv_penalty
                loss_adv_sum += loss_adv.item()
                adv_penalty_sum += adv_penalty.item()
                loss_adv.backward()
                opt_adv.step()
                local_step_adv += 1
                train_logger.add_scalars("loss_adv",
                        {"loss_adv": loss_adv_sum/local_step_adv, 
                         "adv_penalty": adv_penalty_sum/local_step_adv,                                
                        }, global_step)
            
            else:
                # print("step_ae", global_step, local_step_ae)
                loss_1 = model.weight_params[0] * model.loss_fn_1(pred_Y_exp, Y_exp) # normalized weight*loss
                loss_2 = model.weight_params[1] * model.loss_fn_2(pred_Y_exp, Y_exp)
                if global_step == 0:
                    l0_1, l0_2 = loss_1.detach(), loss_2.detach()
                    # print("l0", l0_1, l0_2)
                loss_1_sum += loss_1.item()
                loss_2_sum += loss_2.item()
                # treat current discrminator as the optimal
                loss = torch.div(torch.add(loss_1,loss_2), 2) - model.hparams["reg_adv"]*adv_loss
                loss_sum += loss.item()
                corr_sum_train += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())
                opt_1.zero_grad()
                loss.backward(retain_graph=True) # retain_graph for G1R and G2R
                
                # calculate norm of current gradient w.r.t. 1st layer param W: influence of loss on overall training
                W = list(model.parameters())[0] # not a parameter for Lgrad
                # print("W", W.shape)
                G1R = torch.autograd.grad(loss_1, W, retain_graph=True, create_graph=True) # tuple of len 1, retain for grad(loss_2, W)
                G1 = torch.norm(G1R[0], 2) # norm of the gradient, tensor of []
                G2R = torch.autograd.grad(loss_2, W, retain_graph=True, create_graph=True) # retain for Lgrad.backward()
                G2 = torch.norm(G2R[0], 2)
                G_avg = torch.div(torch.add(G1, G2), 2) 
                
                # calculate relative losses 
                lhat_1 = torch.div(loss_1, l0_1)
                lhat_2 = torch.div(loss_2, l0_2)
                lhat_avg = torch.div(torch.add(lhat_1, lhat_2), 2)
                
                # calculate relative inverse training rates (inv_rate sum to 2): less than 1 -> train faster; larger than 1: slower
                inv_rate_1 = torch.div(lhat_1, lhat_avg)
                inv_rate_2 = torch.div(lhat_2, lhat_avg)
                
                # calculate the constant target 
                C1 = G_avg*(inv_rate_1)**model.hparams["alpha"]
                C2 = G_avg*(inv_rate_2)**model.hparams["alpha"]
                C1 = C1.detach().squeeze() # tensor of []
                C2 = C2.detach().squeeze()

                # IMPORTANT: clean accumulated gradients of loss w.r.t. loss weights
                opt_2.zero_grad() 
                # calculate the gradient loss
                Lgrad = torch.add(model.grad_loss(G1, C1), model.grad_loss(G2, C2)) # sum of L1 loss
                train_logger.add_scalar("Lgrad", Lgrad.detach().item(), global_step)
                # print("orig weights are:", [w.item() for w in model.weight_params]) # raw before updating
                Lgrad.backward()
                
                # update loss weights
                opt_2.step()
                # update the model weights
                opt_1.step()
                
                # normalize the loss weights: sum to 2
                coef = 2/torch.add(model.weight_loss_1, model.weight_loss_2)
                model.weight_params = [coef*model.weight_loss_1, coef*model.weight_loss_2] # update
                # print("normalized updated weights are:", [w.item() for w in model.weight_params])
                local_step_ae += 1
            global_step += 1

        train_logger.add_scalars("loss_ae",
                                {"loss": loss_sum/local_step_ae, 
                                 "loss_1": loss_1_sum/local_step_ae, 
                                 "loss_2": loss_2_sum/local_step_ae,                                
                                }, global_step)
        train_logger.add_scalars("weight_info",
                                {
                                 "normalized_weight_loss_1": model.weight_params[0].item(), 
                                 "normalized_weight_loss_2": model.weight_params[1].item(),                                
                                }, global_step)
        if args.verbose:
            print("(Train) epoch: {:03d}, global_step: {:d}, loss: {:.4f}, loss_adv: {:.4f}, corr: {:.4f}".format(
                epoch, global_step, loss_sum/local_step_ae, loss_adv_sum/local_step_adv if local_step_adv>0 else 0., corr_sum_train/local_step_ae))

        model.eval()
        with torch.no_grad():
            corr_sum_val = 0.
            for sample in val_set:
                X_exp, day, Y_exp = sample
                X_exp, day, Y_exp = model.move_inputs_(X_exp, day, Y_exp)
                pred_Y_exp = model(X_exp, day)
                corr_sum_val += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())
            train_logger.add_scalars("corr_info", 
                                    {"corr_train": corr_sum_train/local_step_ae,
                                     "corr_val": corr_sum_val/len(val_set),
                                    }, global_step)
            if args.verbose:
                print("(Val) epoch: {:03d}, global_step: {:d}, corr: {:.4f}".format(epoch, global_step, corr_sum_val/len(val_set)))

        stop = model.early_stopping(corr_sum_val/len(val_set))
        if stop:
            print(f"early_stop: {epoch}")
            break

    if args.save:
        save_model(model)
        save_hparams(model)
        print("Saved model and hparams")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("-p", "--prep", type=str, default=None)
    parser.add_argument("-o", "--optimizer", type=str, default="SGD")
    # parser.add_argument('--schedule_lr', action = "store_true")
    parser.add_argument("-n", "--n_epochs", type=int, default=30)
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-hp", "--hparams_path", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()
    print(args)
    torch.manual_seed(3559) # TODO
    train(args)
    # python -m src.train --data_dir toy_data --log_dir logdir -n 100 -p standard_1 -v
