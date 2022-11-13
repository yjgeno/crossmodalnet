import torch
import torch.nn.functional as F
import math


class MultiheadAttention(torch.nn.Module):
    """
    MultiheadAttention layer as activation function.
    qkv_proj + o_proj.
    """
    def __init__(self, input_dim, embed_dim, n_head):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % n_head == 0
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        self.qkv_proj = torch.nn.Linear(input_dim, 3*embed_dim) # stack three weights
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        # Transformer initialization
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        """
        Attention output w/o mask.
        """
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        # attention = attn_logits # raw
        attention = F.softmax(attn_logits, dim=-1) # softmax
        values = torch.matmul(attention, v)
        return values, attention
    
    def forward(self, 
                X, 
                mask=None,
                return_attn=False):
        batch_size, seq_length, _ = X.size() # X: [Batch, seq_Len, embed_dim]
        qkv = self.qkv_proj(X) # self-attn

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.n_head, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, head, seq_Len, embed_dim]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine values outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, seq_Len, head, embed_dim]
        values = values.reshape(batch_size, seq_length, self.embed_dim) # squeeze head
        out = self.o_proj(values)

        if return_attn:
            return out, attention
        else:
            return out


class AttentionEncoderBlock(torch.nn.Module):

    def __init__(self, 
                 input_dim = 128, 
                 n_head = 1, 
                 dim_feedforward = 512, 
                 dropout = 0.1,
                 ):
        """
        Args:
            input_dim: Dimension of input.
            n_head: Number of heads to use in the attention block.
            dim_feedforward: Dimensionality of the hidden layer in the MLP.
            dropout: Dropout rate to use in the dropout layers.
        """
        super(AttentionEncoderBlock, self).__init__()
        # Attention layer: set embed_dim=input_dim
        self.self_attn = MultiheadAttention(input_dim, input_dim, n_head)
        # Two-layer MLP
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, dim_feedforward),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feedforward, input_dim)
        )
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, X, mask=None):
        # Attention
        attn_out = self.self_attn(X, mask=mask)
        X = X + self.dropout(attn_out)
        X = self.norm1(X)
        # MLP
        linear_out = self.linear_net(X)
        X = X + self.dropout(linear_out)
        X = self.norm2(X)
        return X


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # print("pe",pe.size())
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
