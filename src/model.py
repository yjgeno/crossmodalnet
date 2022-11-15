
import torch
import torch.nn as nn
from .loss import NBLoss, GaussNLLLoss, NCorrLoss


class MLP(nn.Module):
    """
    A multilayer perceptron class.
    """
    def __init__(self, sizes, batch_norm=True):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[s], sizes[s + 1]),
                nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                nn.ReLU(),
            ]
        layers = [l for l in layers if l is not None][:-1]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Time_block(nn.Module):
    """
    A multilayer perceptron class for time (day).
    """
    def __init__(self, n_out, n_in=4):
        super(Time_block, self).__init__()
        layers = [
                nn.Linear(n_in, 4*n_in),
                nn.BatchNorm1d(4*n_in),
                nn.ReLU(),
                nn.Linear(4*n_in, n_out),
                nn.BatchNorm1d(n_out),
                 ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class multimodal_AE(nn.Module):
    """
    An autoencoder model.
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_latent: int = 512,
        dropout = 0.15,
        alpha: float = 0.5,
        loss_ae: str = "multitask",
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    ):
        super(multimodal_AE, self).__init__()
        self.n_input = n_input 
        self.n_output = n_output # dim
        self.loss_ae = loss_ae
        self.alpha = alpha
        self.device = device

        # AE
        layers = [
                nn.Linear(n_input, n_latent),
                nn.BatchNorm1d(n_latent),
                nn.ReLU(),
                nn.Dropout(p = dropout),
                nn.Linear(n_latent, n_latent),
                nn.BatchNorm1d(n_latent),
                 ]
        self.first_layer = nn.Sequential(*layers)
        self.time_layer = Time_block(n_out = n_latent)
        self.encoder = MLP(
            [n_latent]*2
            + [n_output],
            **kwargs
        )

        # AE loss
        if self.loss_ae == "multitask":
            self.loss_fn_1, self.loss_fn_2 = NCorrLoss(), nn.MSELoss()
            self.grad_loss = nn.L1Loss()
            # self.weight_loss_1 = torch.FloatTensor([1]).clone().detach().requires_grad_(True).to(self.device)
            # self.weight_loss_2 = torch.FloatTensor([1]).clone().detach().requires_grad_(True).to(self.device)
            self.weight_loss_1 = torch.tensor([1.], requires_grad = True, device = self.device)
            self.weight_loss_2 = torch.tensor([1.], requires_grad = True, device = self.device)
            self.weight_params = [self.weight_loss_1, self.weight_loss_2]
        else:
            raise Exception("")

        self.to(self.device) # send model to CUDA


    def move_inputs_(self, *args):
        """
        Move inputs to CUDA.
        """
        return [x.to(self.device) for x in args]


    def forward(self, X, T=None):
        """
        Predict Y given X.
        """
        if T is not None:
            X = self.first_layer(X) + self.time_layer(T)
        else:
            X = self.first_layer(X)
        reconstructions = self.encoder(X)
        return reconstructions



def save_model(model, name: str = "multitask"):
    from torch import save
    import os
    if isinstance(model, multimodal_AE):
        return save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{name}.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(path, **kwargs): # num_genes, num_drugs, loss_ae
    from torch import load
    r = multimodal_AE(**kwargs)
    r.load_state_dict(load(path, map_location='cpu'))
    return r
