from torch import nn
from torchmetrics.functional import pearson_corrcoef, r2_score
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import optim
import torch.nn.functional as F


class NCorrLoss(nn.Module):
    """
    Negative correlation loss.
    """
    def __init__(self):
      super(NCorrLoss, self).__init__()

    @staticmethod
    def tile(y, method:str = "mean"):
      if method == "mean":
        return torch.tile(y.mean(dim=1).unsqueeze(1), (1, y.shape[1]))
      if method == "norm":
        return torch.tile(y.norm(dim=1).unsqueeze(1), (1, y.shape[1]))

    def forward(self, pred_y, y, eps: float = 1e-6):
      pred_n = pred_y - self.tile(pred_y) + eps
      target_n = y - self.tile(y) + eps
      pred_n = pred_n / (self.tile(pred_n, method="norm") + eps)
      target_n = target_n / (self.tile(target_n, method="norm") + eps)
      r = (pred_n * target_n).sum(dim=1).mean() # [-1, 1], reduction="mean"
      r = max((r + 1)/2, torch.tensor(eps,
                        dtype=torch.float)) # [eps, 1]
      loss = (-torch.log(r))**0.15
      return loss


class MLP(nn.Module):
    def __init__(self,
                 n_genes=3000,
                 n_proteins=140,
                 hidden_dims=None,
                 activation="LeakyReLU",
                 dropout_rate=0.2,
                 bs=8,
                 **kwargs):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        self._n_genes = n_genes
        self._n_proteins = n_proteins
        self.bs = bs
        self.metrics = {'R2': r2_score,
                        "pearsonR": pearson_corrcoef}
        self.layers = self.build_layers([n_genes] + hidden_dims + [n_proteins],
                                        activation=activation,
                                        dropout_rate=dropout_rate)

    def build_layers(self,
                     hidden_dims,
                     use_layernorm=False,
                     use_batchnorm=True,
                     use_dropout=True,
                     activation="LeakyReLU",
                     dropout_rate=0.2):
        modules = []
        for i in range(len(hidden_dims) - 1):
            layers = [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), ]
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dims[i + 1]))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(1))
            layers.append(getattr(nn, activation)())

            if use_dropout:
                layers.append(nn.Dropout(p=dropout_rate))
            modules.append(nn.Sequential(*layers))

        return nn.Sequential(*modules)

    def forward(self, X):
        return self.layers(X)

    def calc_metrics(self, pred, targ):
        return {k: v(torch.squeeze(pred).T, torch.squeeze(targ).T)
                for k, v in self.metrics.items()}


class Module(pl.LightningModule):
    def __init__(self, use_ncorr_loss=False, **kwargs):
        super().__init__()
        self.model = MLP(**kwargs)
        self.optim_params = kwargs.get("optim_params",
                                       {"optim": "AdamW", "lr": 1e-3})
        if use_ncorr_loss:
            self.loss = NCorrLoss()
        else:
            self.loss = getattr(F, kwargs.get("loss", "mse_loss"))
        self.mse_loss = F.mse_loss
        self._test_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch["X"].to(torch.float32), batch["y"].to(torch.float32)
        y_pred = self.model(X,
                            )
        loss = self.loss(y_pred, y)
        metrics = self.model.calc_metrics(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({"train_" + k: v.mean() for k, v in metrics.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch["X"].to(torch.float32), batch["y"].to(torch.float32)
        y_pred = self.model(X)
        loss = self.loss(y_pred, y)
        metrics = self.model.calc_metrics(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict({"val_" + k: v.mean() for k, v in metrics.items()})
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch["X"].to(torch.float32), batch["y"].to(torch.float32)
        y_pred = self.model(X)
        loss = self.mse_loss(y_pred, y)
        metrics = self.model.calc_metrics(y_pred, y)
        self._test_outputs.append({"loss": loss, **metrics})
        return {"loss": loss, **metrics}

    def on_test_epoch_end(self) -> None:
        prs = []
        r2s = []
        loss = []
        for output in self._test_outputs:
            prs.append(torch.ravel(output["pearsonR"]))
            r2s.append(torch.ravel(output["R2"]))
            loss.append(torch.ravel(output["loss"]))
        prs = torch.cat(prs)
        r2s = torch.cat(r2s)
        loss = torch.cat(loss)

        self.log("test_pearsonR (mean)", torch.mean(prs))
        self.log("test_pearsonR (std)", torch.std(prs))

        self.log("test_R2 (mean)", torch.mean(r2s))
        self.log("test_R2 (std)", torch.std(r2s))

        self.log("test_mse_loss (mean)", torch.mean(loss))
        self.log("test_mse_loss (std)", torch.std(loss))
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch["X"].to(torch.float32), batch["y"].to(torch.float32)
        y_pred = self.model(X).cpu().numpy()
        metrics = self.model.calc_metrics(y_pred, y)
        return y_pred, metrics

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optim_params.pop("optim"))(self.parameters(), **self.optim_params)
        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]