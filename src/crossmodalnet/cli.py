"""Command-line entry points for CrossmodalNet."""

import argparse

import torch


def _build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crossmodalnet-train", description="Train CrossmodalNet")
    parser.add_argument("-x", "--data-x", dest="data_x", type=str, required=True,
                        help="Path to .h5ad file with X (gene expression).")
    parser.add_argument("-y", "--data-y", dest="data_y", type=str, required=True,
                        help="Path to .h5ad file with Y (protein expression).")
    parser.add_argument("--log-dir", dest="log_dir", type=str, default=None,
                        help="Optional TensorBoard log directory (under ./logger/).")
    parser.add_argument("-p", "--prep", type=str, default=None,
                        help="Preprocessing key: binary | standard_0 | standard_1 | PCA | tSVD | magic | tfidf.")
    parser.add_argument("-o", "--optimizer", type=str, default="SGD", choices=["Adam", "SGD"])
    parser.add_argument("-n", "--n-epochs", dest="n_epochs", type=int, default=30)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("-hp", "--hparams-path", dest="hparams_path", type=str, default=None,
                        help="Path to a hparams JSON file.")
    parser.add_argument("--tkey", type=str, default="day", help="Time column in obs.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--save", action="store_true", help="Save model state_dict and hparams.")
    parser.add_argument("--save-dir", dest="save_dir", type=str, default=".",
                        help="Directory to write saved artifacts (with --save).")
    parser.add_argument("--seed", type=int, default=3559)
    return parser


def train_main(argv=None) -> None:
    from .train import train

    args = _build_train_parser().parse_args(argv)
    print(args)
    torch.manual_seed(args.seed)
    train(args)


def _build_tune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crossmodalnet-tune", description="Run Ray Tune for CrossmodalNet")
    parser.add_argument("-x", "--data-x", dest="data_x", type=str, required=True)
    parser.add_argument("-y", "--data-y", dest="data_y", type=str, required=True)
    parser.add_argument("--tkey", type=str, default="day")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--max-t", dest="max_t", type=int, default=300)
    parser.add_argument("--grace-period", dest="grace_period", type=int, default=10)
    parser.add_argument("--output-csv", dest="output_csv", type=str, default="raytune_results.csv")
    parser.add_argument("--test", action="store_true", help="Quick smoke run.")
    return parser


def tune_main(argv=None) -> None:
    from .tune import run_tune

    args = _build_tune_parser().parse_args(argv)
    run_tune(
        data_path_x=args.data_x,
        data_path_y=args.data_y,
        time_key=args.tkey,
        trials=args.trials,
        test=args.test,
        max_t=args.max_t,
        grace_period=args.grace_period,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    train_main()
