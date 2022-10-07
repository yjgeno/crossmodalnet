import torch
import numpy as np
import torch.nn.functional as F


class NBLoss(torch.nn.Module):
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


class GaussNLLLoss(torch.nn.Module):
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


class NCorrLoss(torch.nn.Module):
    """Negative correlation loss.

    Precondition:
    y_true.mean(axis=1) == 0
    y_true.std(axis=1) == 1

    Returns:
    -1 = perfect positive correlation
    1 = totally negative correlation
    """

    def __init__(self):
        super(NCorrLoss, self).__init__()

    def forward(self, pred_y, y):
        my = torch.mean(pred_y, dim=1)
        my = torch.tile(torch.unsqueeze(my, dim=1), (1, y.shape[1]))
        ym = pred_y - my
        r_num = torch.sum(torch.multiply(y, ym), dim=1)
        r_den = torch.sqrt(
            torch.sum(torch.square(ym), dim=1) * float(y.shape[-1])
        )
        r = torch.mean(r_num / r_den)
        return -r
