
import torch
import torch.nn as nn
from .loss import NCorrLoss
import matplotlib.pyplot as plt


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



class CrossmodalNet(nn.Module):
    """
    An autoencoder model.
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        time_p: list,
        hparams_dict: dict = None,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    ):
        super(CrossmodalNet, self).__init__()
        self.n_input = n_input 
        self.n_output = n_output # dim
        self.device = device
        self.set_hparams(hparams_dict=hparams_dict) # self.hparams

        # AE
        layers = [
                nn.Linear(n_input, self.hparams["n_latent"]),
                nn.BatchNorm1d(self.hparams["n_latent"]),
                nn.ReLU(),
                nn.Dropout(p = self.hparams["first_layer_dropout"]),
                nn.Linear(self.hparams["n_latent"], self.hparams["n_latent"]),
                nn.BatchNorm1d(self.hparams["n_latent"]),
                 ]
        self.first_layer = nn.Sequential(*layers)
        self.encoder = MLP(
            self.hparams["encoder_hidden"]
            + [n_output],
            **kwargs
        )
        self.tau = torch.LongTensor(time_p) # Days
        # self.t_encoder = TimeEncoding(self.hparams["n_latent"])
        # self.t_emb = self.t_encoder(self.tau, fn=False) # emb: [Days, latent]
        self.t_emb = nn.Embedding(len(self.tau), self.hparams["n_latent"])

        # Adv discriminator
        self.adv_mlp = MLP([self.hparams["n_latent"]]
                            + self.hparams["adv_hidden"] 
                            + [len(time_p)])

        # AE loss
        self.loss_fn_1, self.loss_fn_2 = NCorrLoss(), nn.MSELoss()
        self.grad_loss = nn.L1Loss()
        self.weight_loss_1 = torch.tensor([1.], requires_grad = True, device = self.device)
        self.weight_loss_2 = torch.tensor([1.], requires_grad = True, device = self.device)
        self.weight_params = [self.weight_loss_1, self.weight_loss_2]
        # Adv loss
        self.loss_fn_adv = nn.CrossEntropyLoss()

        # params
        self.get_params = lambda model: list(model.parameters())
        self.params_ae = self.get_params(self.first_layer) + self.get_params(self.encoder) + self.get_params(self.t_emb)
        self.params_adv = self.get_params(self.adv_mlp) 

        self.to(self.device) # send model to CUDA

    def set_hparams(self, hparams_dict=None):
        self._hparams = {
            "n_latent": 512,
            "first_layer_dropout": 0.15,
            "encoder_hidden": [512, 512], 
            "adv_hidden": [128],
            "reg_adv": 2,
            "penalty_adv": 2,
            "ae_lr": 7.5e-2,
            "weight_lr": 2.4e-3,
            "adv_lr": 1e-2,
            "ae_wd": 4e-7,
            "adv_wd": 4e-7,
            "adv_step": 3,
            "alpha": 0.5,
        }
        if hparams_dict is not None:
            for key in hparams_dict:
                try:
                    self._hparams[key] = hparams_dict[key]
                except KeyError:
                    continue

    @property
    def hparams(self):
        """
        Returns a list of  hyper-parameters.
        """
        return self._hparams
     
    def move_inputs_(self, *args):
        """
        Move inputs to CUDA.
        """
        return [x.to(self.device) for x in args]

    @staticmethod
    def compute_gradients(output, input):
        grads = torch.autograd.grad(output, input, create_graph=True)
        grads = grads[0].pow(2).mean()
        return grads

    def forward(self, X, T=None, return_latent=False):
        """
        Predict Y given X.
        T: One-hot vectors for time.
        """
        latent_base = self.first_layer(X)
        if T is not None:
            T_idx = T.argmax(1) # [Batch,]
            # print("T shape:", T.shape, T_idx.shape)
            latent = latent_base + self.t_emb(T_idx) # [Batch, latent]
        reconstructions = self.encoder(latent)
        if return_latent:
            return reconstructions, latent_base
        return reconstructions


# class TimeEncoding(nn.Module):
#     def __init__(self, n_out=512):
#         super(TimeEncoding, self).__init__()
#         self.n_out = n_out
#         self.w0 = nn.parameter.Parameter(torch.randn(n_out, 1))
#         self.b0 = nn.parameter.Parameter(torch.randn(n_out, 1))
#         self.fn = torch.exp

#     def forward(self, tau, fn=False):
#         t_len = len(tau)
#         tau = tau.expand(self.n_out, t_len) # expand 0
#         w = self.w0.expand(self.n_out, t_len) # expand 1
#         t_emb = torch.mul(tau, w) + self.b0
#         # print(t_emb.shape)
#         if fn:
#             return self.fn(t_emb.T)
#         return t_emb.T

#     @staticmethod
#     def plot(pe):
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,3))
#         pos = ax.imshow(pe, cmap="Reds", extent=(1,pe.shape[1]+1,pe.shape[0]+1,1))
#         fig.colorbar(pos, ax=ax)
#         ax.set_xlabel("Hidden dimension")
#         ax.set_ylabel("Discrete time")
#         ax.set_xticks([1]+[i for i in range(1,1+pe.shape[1])])
#         ax.set_yticks([1]+[i for i in range(1,1+pe.shape[0])])
#         plt.show()


def save_model(model, name: str = "CrossmodalNet"):
    from torch import save
    import os
    if isinstance(model, CrossmodalNet):
        return save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{name}.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(path, **kwargs): # num_genes, num_drugs, loss_ae
    from torch import load
    r = CrossmodalNet(**kwargs)
    r.load_state_dict(load(path, map_location='cpu'))
    return r
