import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer.
        :param d_internal: The "internal" dimension used for Keys and queries.
        """
        super().__init__()
        self.embed_dim = d_model
        self.qkv_proj = torch.nn.Linear(d_model, 3*d_internal) # stack three weights
        self.o_proj = torch.nn.Linear(d_internal, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        # Transformer initialization
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    @staticmethod
    def scaled_dot_product(q, k, v):
        """
        Attention output w/o mask.
        """
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1) # softmax
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, input_vecs, return_attention= False): # [seq, d_model]
        qkv = self.qkv_proj(input_vecs)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = self.scaled_dot_product(q, k, v)
        out = self.o_proj(values)
        if return_attention:
            return out, attention
        return out

