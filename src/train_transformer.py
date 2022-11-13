import torch
import os
import torch.utils.tensorboard as tb
from .data import sc_Dataset_index, collator_fn, load_data
from .model import Transformer_multi, save_model
from .utils import corr_score


def train(args):
    """
    Train model.
    """
    # load data

    data_path_X = os.path.join(args.data_dir, "multi_train_x.h5ad")
    data_path_Y = os.path.join(args.data_dir, "multi_train_y.h5ad")
    dataset = sc_Dataset_index(data_path_X, data_path_Y)
    train_set, val_set = load_data(dataset, batch_size = args.batch_size, collate_fn=collator_fn)

    # init model
    model = Transformer_multi(num_positions=dataset.n_feature_X, 
                              n_output=dataset.n_feature_Y,
                              )
    print(model)

    # optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = 1e-5)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = 5e-4)
    
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
            X_indexed, Y_exp, padd_mask = sample
            X_indexed, Y_exp, padd_mask =  model.move_inputs_(X_indexed, Y_exp, padd_mask)
            optimizer.zero_grad()
            pred_Y_exp = model(X_indexed, padd_mask=padd_mask)
            # print(type(pred_Y_exp), len(pred_Y_exp), type(Y_exp), len(Y_exp))
            loss = model.loss_fn_ae(pred_Y_exp, Y_exp)
            train_logger.add_scalar("loss", loss.item(), global_step)
            loss_sum += loss.item()
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
                X_indexed, Y_exp, padd_mask = sample
                X_indexed, Y_exp, padd_mask =  model.move_inputs_(X_indexed, Y_exp, padd_mask)
                pred_Y_exp = model(X_indexed, padd_mask=padd_mask)
                # loss = loss_fn_ae(pred_Y_exp, Y_exp)
                # valid_logger.add_scalar('loss', loss.item(), global_step)
                corr_sum_val += corr_score(Y_exp.detach().cpu().numpy(), pred_Y_exp.detach().cpu().numpy())
            train_logger.add_scalars("corr_info", 
                                    {"corr_train": corr_sum_train/len(train_set),
                                     "corr_val": corr_sum_val/len(val_set),
                                    }, global_step)
            if args.verbose:
                print("(Val) epoch: {:03d}, global_step: {:d}, corr: {:.4f}".format(epoch, global_step, corr_sum_val/len(val_set)))

    if args.save:
        save_model(model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("-l", "--loss_ae", type=str, default="mse")
    parser.add_argument("-o", "--optimizer", type=str, default="Adam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-n", "--n_epochs", type=int, default=30)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()
    print(args)
    torch.manual_seed(6869) # TODO
    train(args)
    # python -m src.train_transformer --data_dir toy_data --log_dir trans_multi -l ncorr -n 30 -v
