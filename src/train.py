import torch
import os
import torch.utils.tensorboard as tb
from .data import sc_Dataset, load_data
from .model import CITE_AE, MULTIOME_AE, save_model, load_model
from .utils import corr_score


def train(args):
    """
    Train model.
    """
    # load data
    if args.mode == "CITE":
            data_path_X = os.path.join(args.data_dir, "cite_train_x.h5ad")
            data_path_Y = os.path.join(args.data_dir, "cite_train_y.h5ad")
    if args.mode == "MULTIOME":
            data_path_X = os.path.join(args.data_dir, "multi_train_x.h5ad")
            data_path_Y = os.path.join(args.data_dir, "multi_train_y.h5ad")
    dataset = sc_Dataset(
            data_path_X = data_path_X,
            data_path_Y = data_path_Y,
            time_key = "day",
            celltype_key = "cell_type",
            )
    train_set, val_set = load_data(dataset)

    # init model
    if args.mode == "CITE":
        model = CITE_AE(n_input = dataset.n_feature_X, 
                            n_output= dataset.n_feature_Y,
                            loss_ae = args.loss_ae,
                            )
    if args.mode == "MULTIOME":
        model = MULTIOME_AE(chrom_len_dict = dataset.chrom_len_dict, 
                              chrom_idx_dict = dataset.chrom_idx_dict,
                              n_output= dataset.n_feature_Y,
                              loss_ae = args.loss_ae,
                             )
    print(model)

    # optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = 1e-5)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = 5e-4)
    if args.schedule_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 5)
    
    # logging
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(os.path.join("logger", args.log_dir, "train"), flush_secs=1)
        # valid_logger = tb.SummaryWriter(os.path.join("logger", args.log_dir, 'valid'), flush_secs=1)
    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        loss_sum, corr_sum_train = 0., 0.
        for sample in train_set:
            X_exp, day, celltype, Y_exp = sample
            X_exp, day, celltype, Y_exp =  model.move_inputs_(X_exp, day, celltype, Y_exp)
            optimizer.zero_grad()
            pred_Y_exp = model(X_exp)
            loss = model.loss_fn_ae(pred_Y_exp, Y_exp)
            train_logger.add_scalar("loss", loss.item(), global_step)
            loss_sum += loss.item()
            if model.loss_ae in model.loss_type2:
                    pred_Y_exp = model.sample_pred_from(pred_Y_exp)
            corr_sum_train += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            global_step += 1
        if args.verbose:
            print("(Train) epoch: {:03d}, global_step: {:d}, loss: {:.4f}, corr: {:.4f}".format(epoch, global_step, loss_sum/len(train_set), corr_sum_train/len(train_set)))

        model.eval()
        with torch.no_grad():
            corr_sum_val = 0.
            for sample in val_set:
                X_exp, day, celltype, Y_exp = sample
                X_exp, day, celltype, Y_exp =  model.move_inputs_(X_exp, day, celltype, Y_exp)
                pred_Y_exp = model(X_exp)
                # loss = loss_fn_ae(pred_Y_exp, Y_exp)
                # valid_logger.add_scalar('loss', loss.item(), global_step)
                if model.loss_ae in model.loss_type2:
                    pred_Y_exp = model.sample_pred_from(pred_Y_exp)
                corr_sum_val += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())
            train_logger.add_scalars("corr_info", 
                                    {"corr_train": corr_sum_train/len(train_set),
                                     "corr_val": corr_sum_val/len(val_set),
                                    }, global_step)
            if args.verbose:
                print("(Val) epoch: {:03d}, global_step: {:d}, corr: {:.4f}".format(epoch, global_step, corr_sum_val/len(val_set)))

        if args.schedule_lr:
            train_logger.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step)
            scheduler.step(corr_sum_val/len(val_set)) # update according to valid set

    if args.save:
        save_model(model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--data_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("-M", "--mode", type=str, default="CITE")
    parser.add_argument("-L", "--loss_ae", type=str, default="mse")
    parser.add_argument("-O", "--optimizer", type=str, default="Adam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument('--schedule_lr', action = "store_true")
    parser.add_argument("-N", "--n_epochs", type=int, default=30)
    parser.add_argument("-B", "--batch_size", type=int, default=256)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()
    print(args)
    torch.manual_seed(6869) # TODO
    train(args)
    # python -m src.train --data_dir toy_data --log_dir log_cite -N 30 -v
    # python -m src.train --data_dir toy_data --log_dir log_multi -M MULTIOME -L ncorr -N 30 -v
