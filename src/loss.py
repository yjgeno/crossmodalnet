import torch
import numpy as np


class NBLoss(torch.nn.Module):
    """Negative binomial negative log-likelihood.

    """
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, mu, y, theta, eps=1e-8):
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
        super().__init__()

    def forward(self, preds, targets):
        my = torch.mean(preds, dim=1)
        my = torch.tile(torch.unsqueeze(my, dim=1), (1, targets.shape[1]))
        ym = preds - my
        r_num = torch.sum(torch.multiply(targets, ym), dim=1)
        r_den = torch.sqrt(
            torch.sum(torch.square(ym), dim=1) * float(targets.shape[-1])
        )
        r = torch.mean(r_num / r_den)
        return -r

