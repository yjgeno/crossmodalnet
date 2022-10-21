
import torch
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
        loss_ae: str = "multitask",
        hparams_dict: dict = None,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(multimodal_AE, self).__init__()
        self.n_input = n_input 
        self.n_output = n_output # dim
        self.loss_ae = loss_ae
        self.device = device

        # set hyperparameters
        self.set_hparams(hparams_dict = hparams_dict)

        # AE
        self.encoder = MLP(
            [n_input]
            + self.hparams["autoencoder_width"]
            + [self.hparams["latent_dim"]]
        )

        self.decoder = MLP(
            [self.hparams["latent_dim"]]
            # + self.hparams["autoencoder_width"]
            + [n_output] # *2
        )

        # AE loss
        if self.loss_ae == "multitask":
            self.loss_fn_1, self.loss_fn_2 = NCorrLoss(), torch.nn.MSELoss()
            self.grad_loss = torch.nn.L1Loss()
            # self.weight_loss_1 = torch.FloatTensor([1]).clone().detach().requires_grad_(True).to(self.device)
            # self.weight_loss_2 = torch.FloatTensor([1]).clone().detach().requires_grad_(True).to(self.device)
            self.weight_loss_1 = torch.tensor([1.], requires_grad = True, device = self.device)
            self.weight_loss_2 = torch.tensor([1.], requires_grad = True, device = self.device)
            self.weight_params = [self.weight_loss_1, self.weight_loss_2]
        else:
            raise Exception("")

        self.to(self.device) # send model to CUDA


    def set_hparams(self, hparams_dict: dict = None):
        self._hparams = {
            "latent_dim": 32,
            "autoencoder_width": [512, 128],
            "alpha": 1.,
        }  # set default
        if hparams_dict is not None: 
            for key in hparams_dict:
                try:
                    self._hparams[key] = hparams_dict[key] 
                except KeyError:
                    continue


    @property
    def hparams(self):
        """
        Returns a list of the hyper-parameters.
        """
        return self._hparams


    # @hparams.setter
    # def hparams(self, hparams_dict: dict):
    #     self._hparams = hparams_dict

    def move_inputs_(self, *args):
        """
        Move inputs to CUDA.
        """
        return [x.to(self.device) for x in args]


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

        if return_latent:
            return reconstructions, latent_basal
        return reconstructions



def save_model(model, name: str = "multitask"):
    from torch import save
    import os
    if isinstance(model, multimodal_AE):
        return save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{name}.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(name: str = "multitask", **kwargs): # num_genes, num_drugs, loss_ae
    from torch import load
    import os
    r = multimodal_AE(**kwargs)
    r.load_state_dict(load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{name}.th'), map_location='cpu'))
    return r
