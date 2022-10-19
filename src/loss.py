import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Loss_(nn.Module):
    def forward(self):
        raise Exception("Implement in subclasses")


class NBLoss(Loss_):
    """Negative binomial negative log-likelihood.

    """
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, pred_y, y, eps=1e-8):
        """Negative binomial negative log-likelihood.

        Args:
            pred_y: Reconstructed means and variances in shape n*2d.
            y: True label in shape n*d.
            eps: for numerical stability, the minimum estimated variance is recommended 
            greater than a small number. Defaults to 1e-8.
        """
        dim = pred_y.shape[1]//2
        mu, theta = pred_y[:, :dim], pred_y[:, dim:]
        if theta.ndimension() == 1:
            theta = theta.view(1, theta.size(0))
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + y * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1)
        )
        res = torch.where(torch.isnan(res), torch.zeros_like(res) + np.inf, res)
        return -torch.mean(res)


class GaussNLLLoss(Loss_):
    """
    Adopted from torch.nn.GaussianNLLLoss.
    Eq.(10) from "Estimating the Mean and Variance of the Target Probability Distribution".
    """
    def __init__(self, full: bool = False, reduction: str = "mean"):
        super(GaussNLLLoss, self).__init__()
        self.full = full
        self.reduction = reduction

    def forward(self, pred_y, y, eps: float = 1e-6):
        dim = pred_y.shape[1]//2
        mu, theta = pred_y[:, :dim], pred_y[:, dim:]
        return F.gaussian_nll_loss(mu, y, theta, full=self.full, eps=eps, reduction=self.reduction)


class NCorrLoss(Loss_):
    """Negative correlation loss.

    """

    def __init__(self):
        super(NCorrLoss, self).__init__()

    @staticmethod
    def tile(y, method:str = "mean"):
        if method == "mean":
            return torch.tile(y.mean(dim=1).unsqueeze(1), (1, y.shape[1]))
        if method == "norm":
            return torch.tile(y.norm(dim=1).unsqueeze(1), (1, y.shape[1]))

    def forward(self, pred_y, y):
        pred_n = pred_y - self.tile(pred_y)
        target_n = y - self.tile(y)
        pred_n = pred_n / self.tile(pred_n, method="norm")
        target_n = target_n / self.tile(target_n, method="norm")
        r = (pred_n * target_n).sum(dim=1)
        r = r.mean()
        # z = torch.cat([pred_y, y], dim=0)
        # r = torch.corrcoef(z)[:pred_y.shape[0], pred_y.shape[0]:].diagonal().mean() # off-diagnoal corr
        # print("r:", r)
        return 1-r
