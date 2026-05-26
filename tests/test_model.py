import os

import numpy as np
import torch

from crossmodalnet.model import CrossmodalNet, load_hparams, save_hparams, save_model


def _make_model(n_input=32, n_output=8, time_p=(2, 3, 4, 7)):
    return CrossmodalNet(
        n_input=n_input,
        n_output=n_output,
        time_p=list(time_p),
        hparams_dict={"n_latent": 16, "encoder_hidden": [16, 16], "adv_hidden": [8]},
    )


def test_forward_output_shape():
    model = _make_model()
    model.eval()
    batch = 4
    x = torch.randn(batch, 32)
    t = torch.eye(4)[torch.randint(0, 4, (batch,))]
    with torch.no_grad():
        y = model(x, t)
    assert y.shape == (batch, 8)


def test_forward_return_latent_shape():
    model = _make_model()
    model.eval()
    batch = 3
    x = torch.randn(batch, 32)
    t = torch.eye(4)[torch.randint(0, 4, (batch,))]
    with torch.no_grad():
        y, latent = model(x, t, return_latent=True)
    assert y.shape == (batch, 8)
    assert latent.shape == (batch, 16)


def test_save_and_load_hparams(tmp_path):
    model = _make_model()
    save_hparams(model, path=str(tmp_path))
    loaded = load_hparams(os.path.join(tmp_path, "hparams.json"))
    assert loaded["n_latent"] == model.hparams["n_latent"]


def test_save_model_writes_file(tmp_path):
    model = _make_model()
    save_model(model, path=str(tmp_path), name="m")
    assert (tmp_path / "m.th").is_file()
