import torch
import torch.nn.functional as F
from .loss import NBLoss, GaussNLLLoss, NCorrLoss
from .activation import MultiheadAttention, AttentionEncoderBlock


class MLP(torch.nn.Module):
    """
    A multilayer perceptron class.
    """
    def __init__(self, sizes, batch_norm=False, dropout=0.):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(), # TODO
                torch.nn.Dropout(p = dropout)
            ]
        layers = [l for l in layers if l is not None][:-2] # last layer
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AE(torch.nn.Module):
    """
    An autoencoder class.
    """
    def __init__(self,        
                 loss_ae: str = "mse",
                 mode: str = "CITE",
                 hparams_dict: dict = None,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        super(AE, self).__init__()
        self.loss_ae = loss_ae
        self.loss_type1, self.loss_type2 = ["mse", "L1", "ncorr"], ["nb", "gauss"]
        self.mode = mode
        self.device = device
        self.relu = torch.nn.ReLU()

        # AE loss
        if self.loss_ae == "nb":
            self.loss_fn_ae = NBLoss()
        elif self.loss_ae == "gauss":
            self.loss_fn_ae = GaussNLLLoss()
        elif self.loss_ae == "ncorr":
            self.loss_fn_ae = NCorrLoss()
        elif self.loss_ae == "mse":
            self.loss_fn_ae = torch.nn.MSELoss()
        elif self.loss_ae == "L1":
            self.loss_fn_ae = torch.nn.L1Loss()
        elif self.loss_ae == "poisson":
            self.loss_fn_ae = torch.nn.PoissonNLLLoss(log_input=False)
        elif self.loss_ae == "comb":
            self.loss_fn_ae_1 = NCorrLoss()
            self.loss_fn_ae_2 = torch.nn.PoissonNLLLoss()
        else:
            raise Exception("")

        # set hyperparameters
        self.set_hparams(hparams_dict=hparams_dict)

    def set_hparams(self, hparams_dict: dict = None):
        if self.mode == "CITE":
            self._hparams = {
                "latent_dim": 32,
                "autoencoder_width": [512, 128],
            }  # set default
        if self.mode == "MULTIOME":
            self._hparams = {
                "latent_dim": 128,
                "autoencoder_width": [4096, 4096, 4096],
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
        
    def move_inputs_(self, *args):
        """
        Move inputs to GPU.
        """
        return [x.to(self.device) for x in args]

    def forward(self):
        raise NotImplementedError("Implement forward in subclass.")

    def sample_pred_from(self, reconstructions):
        if not self.loss_ae in self.loss_type2:
            raise ValueError("")
        else:
            pred_means, pred_vars = (
                reconstructions[:, :reconstructions.size(1)//2],
                reconstructions[:, reconstructions.size(1)//2:],
            )
            if self.loss_ae == "gauss":
                pred_means = torch.normal(mean=pred_means, std=pred_vars)
                pred_means = torch.clamp(pred_means, min=0., max=1e4) # neg in protein counts
            if self.loss_ae == "nb":
                pred_means = torch.clamp(pred_means, min=0., max=1e4)  # TODO
            return pred_means 


class CITE_AE(AE):
    """
    An autoencoder model for CITE data.
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        loss_ae: str = "mse",
        hparams_dict: dict = None,
    ):
        super(CITE_AE, self).__init__(loss_ae, "CITE", hparams_dict)
        self.n_input = n_input
        self.n_output = n_output  # dim
        if loss_ae in self.loss_type2:
            n_output = n_output * 2

        # AE
        self.encoder = MLP(
            [n_input] + self.hparams["autoencoder_width"] + [self.hparams["latent_dim"]]
        )
        self.decoder = MLP(
            [self.hparams["latent_dim"]]
            # + list(reversed(self.hparams["autoencoder_width"]))
            + [n_output]  # *2
        )
        self.to(self.device)

    def forward(
        self,
        X,
        return_latent: bool = False,
    ):
        """
        Predict Y given X.
        """
        latent_basal = self.encoder(X)
        reconstructions = self.decoder(latent_basal)

        # dim = X_reconstructions.shape[1]//2 # self.n_output
        if self.loss_ae == "gauss":
            # convert variance estimates to a positive value in [1e-3, inf)
            pred_means = reconstructions[:, : self.n_output]
            pred_vars = F.softplus(reconstructions[:, self.n_output :]).add(
                1e-3
            )  # constrain positive var
            # X_vars = reconstructions[:, dim:].exp().add(1).log().add(1e-3)
            reconstructions = torch.cat([pred_means, pred_vars], dim=1)

        if self.loss_ae == "nb":
            pred_means = F.softplus(reconstructions[:, : self.n_output]).add(1e-3)
            pred_vars = F.softplus(reconstructions[:, self.n_output :]).add(1e-3)
            # reconstructions[:, :dim] = torch.clamp(reconstructions[:, :dim], min=1e-4, max=1e4)
            # reconstructions[:, dim:] = torch.clamp(reconstructions[:, dim:], min=1e-4, max=1e4)
            reconstructions = torch.cat([pred_means, pred_vars], dim=1)

        if return_latent:
            return reconstructions, latent_basal
        return reconstructions


class MULTIOME_ENCODER(torch.nn.Module):
    """
    An encoder class for ATAC.
    """

    def __init__(self, chrom_len_dict, chrom_idx_dict, d_model, att: bool = False):
        super(MULTIOME_ENCODER, self).__init__()
        self.chrom_len_dict = chrom_len_dict
        self.chrom_idx_dict = chrom_idx_dict
        self.n_chrom = len(chrom_len_dict)
        self.chrom_encoders = {}
        for chrom in chrom_len_dict.keys():  # same key order
            self.chrom_encoders[chrom] = MLP(
                [chrom_len_dict[chrom]] + [d_model]*2  # latent
            )
        # self.joint_encoder = MLP([sizes[0] * self.n_chrom] + sizes[1:] + [sizes[0]])
        self.attention = att
        if self.attention:
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model = d_model, 
                                                             nhead = 4, 
                                                             dim_feedforward = 1024,
                                                             batch_first = False)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers = 6)

    def forward(self, x):
        x_breaks = [
            self.chrom_encoders[chrom](
                x[:, self.chrom_idx_dict[chrom][0] : self.chrom_idx_dict[chrom][1] + 1]
            )
            for chrom in self.chrom_idx_dict.keys()
        ]
        # print([i.shape for i in x_breaks]) # Batch * latent_dim, len = chrom
        if self.attention:
            x_breaks = torch.stack(x_breaks)#.permute(1, 0, 2) # chrom (n_seq) * Batch * latent_dim (d_model)
            x_breaks = self.transformer_encoder(x_breaks)
        x_cat = torch.cat([*x_breaks], dim=1)  # concat all to joint encoder: Batch * (latent_dim*chrom)
        # print("x_cat", x_cat.shape)
        return x_cat


class MULTIOME_AE(AE):
    """
    An autoencoder model for multiome data.
    """
    def __init__(
        self,
        chrom_len_dict,
        chrom_idx_dict,
        n_output: int,
        loss_ae: str = "mse",
        hparams_dict: dict = None,
        **kwargs
    ):
        super(MULTIOME_AE, self).__init__(loss_ae, "MULTIOME", hparams_dict)
        self.n_output = n_output  # dim
        if loss_ae in self.loss_type2:
            n_output = n_output * 2

        # set hyperparameters
        self.set_hparams(hparams_dict=hparams_dict)

        # AE
        self.encoder = MULTIOME_ENCODER(
            chrom_len_dict,
            chrom_idx_dict,
            d_model=self.hparams["latent_dim"],
            **kwargs
        )
        self.decoder = MLP(
            [self.encoder.n_chrom * self.hparams["latent_dim"]]
            + self.hparams["autoencoder_width"]
            + [n_output]
        )
        self.move_inputs_(*list(self.encoder.chrom_encoders.values())) # send each chrom MLP to GPU
        self.to(self.device)

    def forward(
        self,
        X,
        return_latent: bool = False,
        relu_last: bool = False,
    ):
        """
        Predict Y given X.
        """
        latent_basal = self.encoder(X)
        reconstructions = self.decoder(latent_basal)

        if self.loss_ae == "gauss":
            pred_means = reconstructions[:, : self.n_output]
            pred_vars = F.softplus(reconstructions[:, self.n_output :]).add(1e-3) 
            reconstructions = torch.cat([pred_means, pred_vars], dim=1)

        if self.loss_ae == "nb":
            pred_means = F.softplus(reconstructions[:, : self.n_output]).add(1e-3)
            pred_vars = F.softplus(reconstructions[:, self.n_output :]).add(1e-3)
            reconstructions = torch.cat([pred_means, pred_vars], dim=1)

        if relu_last:
            reconstructions = self.relu(reconstructions)
        if return_latent:
            return reconstructions, latent_basal
        return reconstructions


class MULTIOME_DECODER(AE):
    """
    A MLP class for processed low-dimensional ATAC.
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        loss_ae: str = "mse",
        hparams_dict: dict = None,
    ):
        super(MULTIOME_DECODER, self).__init__(loss_ae, "MULTIOME", hparams_dict)
        self.n_input = n_input
        self.n_output = n_output  
        self.decoder = MLP(
            [n_input]
            + list(reversed(self.hparams["autoencoder_width"]))
            + [n_output] 
        )
        self.to(self.device)

    def forward(self, X, relu_last: bool = False):
        if relu_last:
            return self.relu(self.decoder(X))
        return self.decoder(X)


def save_model(model, name: str = "multimodal"):
    from torch import save
    import os
    if isinstance(model, (CITE_AE, MULTIOME_AE, MULTIOME_DECODER)):
        return save(
            model.state_dict(),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.th"),
        )
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(name: str = "multimodal", **kwargs):  # num_genes, num_drugs, loss_ae
    from torch import load
    import os
    try:
        r = CITE_AE(**kwargs)
    except Exception:
        r = MULTIOME_AE(**kwargs)
    r.load_state_dict(
        load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.th"),
            map_location="cpu",
        )
    )
    return r
