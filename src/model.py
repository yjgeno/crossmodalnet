
import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    """
    A multilayer perceptron class.
    """
    def __init__(self, sizes, batch_norm=True):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]
        layers = [l for l in layers if l is not None][:-1]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class multimodal_AE(torch.nn.Module):
    """
    An autoencoder model.
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
    ):
        super(multimodal_AE, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        # set hyperparameters
        self.set_hparams() # self.hparams

        # AE
        self.encoder = MLP(
            [n_input]
            + self.hparams["autoencoder_width"]
            + [self.hparams["latent_dim"]]
        )

        self.decoder = MLP(
            [self.hparams["latent_dim"]]
            + self.hparams["autoencoder_width"]
            + [n_output] # *2
        )


    def set_hparams(self):
        self._hparams = {
            "latent_dim": 128,
            "autoencoder_width": [512, 256],
        }
        return self._hparams


    @property
    def hparams(self):
        """
        Returns a list of the hyper-parameters.
        """
        return self.set_hparams()


    def forward(
        self,
        X,
        return_latent:bool = False,
    ):
        """
        Predict Y given X.
        """
        latent_basal = self.encoder(X)
        X_reconstructions = self.decoder(latent_basal)

        if return_latent:
            return X_reconstructions, latent_basal
        return X_reconstructions


def save_model(model, name: str = "multimodal"):
    from torch import save
    import os
    if isinstance(model, multimodal_AE):
        return save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{name}.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(name: str = "multimodal", **kwargs): # num_genes, num_drugs, loss_ae
    from torch import load
    import os
    r = multimodal_AE(**kwargs)
    r.load_state_dict(load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{name}.th'), map_location='cpu'))
    return r
