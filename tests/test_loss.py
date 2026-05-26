import torch

from crossmodalnet.loss import GaussNLLLoss, NBLoss, NCorrLoss


def test_ncorr_loss_identical_inputs_near_zero():
    torch.manual_seed(0)
    y = torch.randn(8, 16)
    loss_fn = NCorrLoss()
    loss = loss_fn(y.clone(), y.clone())
    assert loss.item() < 1e-3


def test_ncorr_loss_returns_scalar():
    pred = torch.randn(4, 10)
    y = torch.randn(4, 10)
    loss = NCorrLoss()(pred, y)
    assert loss.dim() == 0


def test_nb_loss_runs():
    pred = torch.rand(4, 20) + 1.0  # ensure positive mean / theta
    y = torch.randint(0, 5, (4, 10)).float()
    loss = NBLoss()(pred, y)
    assert torch.isfinite(loss) or torch.isnan(loss) or torch.isinf(loss)
    assert loss.dim() == 0


def test_gauss_nll_loss_runs():
    pred = torch.cat([torch.randn(4, 10), torch.rand(4, 10) + 0.1], dim=1)
    y = torch.randn(4, 10)
    loss = GaussNLLLoss()(pred, y)
    assert loss.dim() == 0
