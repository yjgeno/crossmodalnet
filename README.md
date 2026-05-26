# CrossmodalNet

Interpretable modeling of time-resolved single-cell gene–protein expression.
[Preprint](https://www.biorxiv.org/content/10.1101/2023.05.16.541011v2)

<p align="center">
    <img src="crossmodal_v1.png" alt="drawing" width="300"/>
</p>

## Install

```shell
pip install crossmodalnet
```

Optional extras:

```shell
pip install "crossmodalnet[tune]"     # Ray Tune hyperparameter search
pip install "crossmodalnet[magic]"    # MAGIC imputation preprocessing
pip install "crossmodalnet[viz]"      # mycolorpy palettes for saliency plots
pip install "crossmodalnet[dev]"      # tests + lint + build tooling
```

Requires Python 3.10+.

## Usage

### Training (CLI)

```shell
crossmodalnet-train \
    -x path/to/cite_train_x.h5ad \
    -y path/to/cite_train_y_norm.h5ad \
    --log-dir my_run \
    -o Adam -n 500 -v --save --save-dir ./out
```

### Training (Python)

```python
import torch
from crossmodalnet import CrossmodalNet, sc_Dataset, load_data

dataset = sc_Dataset(
    data_path_X="cite_train_x.h5ad",
    data_path_Y="cite_train_y_norm.h5ad",
    time_key="day",
)
train_loader, val_loader = load_data(dataset, batch_size=256)

model = CrossmodalNet(
    n_input=dataset.n_feature_X,
    n_output=dataset.n_feature_Y,
    time_p=dataset.unique_day,
)
```

### Hyperparameter tuning

```shell
pip install "crossmodalnet[tune]"
crossmodalnet-tune -x x.h5ad -y y.h5ad --trials 50
```

### Example notebook

See [`nb/testing_.ipynb`](nb/testing_.ipynb) for a post-training analysis example.

## License

(c) Yongjian Yang, Texas A&M University. MIT-licensed — see [LICENSE](LICENSE).
