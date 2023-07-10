import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import pearson_corrcoef, r2_score


class Net(nn.Module):
    """
    Line 124~142 + refactor it a little bit
    """
    def __init__(self, n_genes, n_proteins):
        super(Net, self).__init__()
        self._n_proteins = n_proteins
        self._n_genes = n_genes
        self.metrics = {'R2': r2_score,
                        "pearsonR": pearson_corrcoef}

        self.fc1 = nn.Linear(n_genes, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 64 * n_proteins)
        self.fc4 = nn.Conv1d(n_proteins, n_proteins, 64, groups=n_proteins)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)).reshape((-1, self._n_proteins, 64))
        x = F.relu(self.fc4(x))
        return x
    
    def calc_metrics(self, pred, targ):
        return {k: v(torch.squeeze(pred).T, torch.squeeze(targ).T)
                for k, v in self.metrics.items()}


class cTPnetModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Net(**kwargs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001,amsgrad=True, weight_decay=0.001)
        self.loss = nn.MSELoss()
        self._test_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch["X"].to(torch.float32), batch["y"].to(torch.float32)
        y_pred = self.model(X)
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
        loss = self.loss(y_pred, y)
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
        optimizer = optim.Adam(self.model.parameters(), lr=0.001,amsgrad=True, weight_decay=0.001)
        return [optimizer]