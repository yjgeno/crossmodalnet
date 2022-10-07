
import torch
import torch.nn.functional as F
from .loss import NBLoss, GaussNLLLoss, NCorrLoss


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
        loss_ae: str = "mse",
    ):
        super(multimodal_AE, self).__init__()
        self.n_input = n_input 
        self.n_output = n_output # dim
        self.loss_ae = loss_ae
        self.loss_type1, self.loss_type2 = ["mse", "ncorr"], ["nb", "gauss", "custom_"]
        if loss_ae in self.loss_type2:
            n_output = n_output * 2

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

        # AE loss
        if self.loss_ae == "nb":
            self.loss_fn_ae = NBLoss()
        elif self.loss_ae == "gauss":
            self.loss_fn_ae = GaussNLLLoss()
        elif self.loss_ae == "ncorr":
            self.loss_fn_ae = NCorrLoss()
        elif self.loss_ae == "mse":
            self.loss_fn_ae = torch.nn.MSELoss()
        elif self.loss_ae == "custom_":
            self.loss_fn_mse, self.loss_fn_ncorr, self.loss_fn_gauss = torch.nn.MSELoss(), NCorrLoss(), GaussNLLLoss()
        else:
            raise Exception("")


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
        reconstructions = self.decoder(latent_basal)

        # dim = X_reconstructions.shape[1]//2 # self.n_output
        if self.loss_ae == "gauss":
            # convert variance estimates to a positive value in [1e-3, inf)        
            pred_means = reconstructions[:, :self.n_output]
            pred_vars = F.softplus(reconstructions[:, self.n_output:]).add(1e-3) # constrain positive var
            # X_vars = reconstructions[:, dim:].exp().add(1).log().add(1e-3)
            reconstructions = torch.cat([pred_means, pred_vars], dim = 1)

        if self.loss_ae == "nb":
            pred_means = F.softplus(reconstructions[:, :self.n_output]).add(1e-3)
            pred_vars = F.softplus(reconstructions[:, self.n_output:]).add(1e-3)
            # reconstructions[:, :dim] = torch.clamp(reconstructions[:, :dim], min=1e-4, max=1e4)
            # reconstructions[:, dim:] = torch.clamp(reconstructions[:, dim:], min=1e-4, max=1e4)
            reconstructions = torch.cat([pred_means, pred_vars], dim = 1)

        if self.loss_ae == "custom_": # TODO
            pred_means = reconstructions[:, :self.n_output]
            pred_vars = F.softplus(reconstructions[:, self.n_output:]).add(1e-3)
            reconstructions = torch.cat([pred_means, pred_vars], dim = 1)

        if return_latent:
            return reconstructions, latent_basal
        return reconstructions


    def sample_pred_from(self, reconstructions):
        if not self.loss_ae in self.loss_type2:
            raise ValueError("")
        else:
            pred_means, pred_vars = reconstructions[:, :self.n_output], reconstructions[:, self.n_output:]
            if self.loss_ae == "gauss":
                pred_means = torch.normal(mean = pred_means, std = pred_vars)
                # pred_means = torch.clamp(pred_means, min=0., max=1e4) # neg in protein counts 
            if self.loss_ae == "nb":
                pass # TODO
            if self.loss_ae == "custom_":
                pred_means = torch.normal(mean = pred_means, std = pred_vars)
                # print("custom", torch.sum(pred==0)/(pred.shape[0]*pred.shape[1]))
                # pred_means = torch.clamp(pred_means, min=0., max=1e4) # neg in protein counts 
            return pred_means # TODO



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
