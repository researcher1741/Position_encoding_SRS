#  # PyTorch
import torch.nn as nn
from torch import Tensor

from src.Attention_blocks import MHA
from src.Encoding import IdentityEncoding


class FF(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x


class PointWiseFeedForward(FF):
    """ Feedfordward network on top of the Encoder"""

    def __init__(self, config):
        super().__init__()
        config.hidden_act = config.hidden_act.lower()
        self.conv1 = nn.Conv1d(config.hidden_units, config.hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=config.hidden_dropout_prob)
        self.act = Activation(config.hidden_act, config.maxlen)
        self.conv2 = nn.Conv1d(config.hidden_units, config.hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=config.hidden_dropout_prob)
        # Init
        self._init_weights(self.conv1)
        self._init_weights(self.conv2)

    def forward(self, inputs):
        # as Conv1D requires (N, C, Length)
        outputs = inputs.transpose(-1, -2).contiguous()
        outputs = self.conv1(outputs)
        outputs = self.act(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        outputs = outputs.transpose(-1, -2).contiguous()
        outputs += inputs
        return outputs


class PointWiseFeedForwardOut(FF):
    """
    Feedfordward network on top of the Decoder. Note that the sigmoid function is applied directly on the output.
    """
    def __init__(self, config):
        super().__init__()
        self.ffn = nn.Linear(in_features=config.embedding_d, out_features=1)
        self._init_weights(self.ffn)

    def forward(self, s):
        y = self.ffn.forward(s, )
        y = y.squeeze()  # Squeeze output ([B, L, 1] -> [B, L])
        return y


class UserBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Architecture
        self.attn_layer = MHA(config.embedding_d,
                              config.num_heads,
                              config.attention_probs_dropout_prob,
                              config.intercalate_act)
        if config.user_FF:
            self.ffn = nn.Linear(in_features=config.embedding_d,
                                 out_features=config.embedding_d)

        if config.user_act:
            self.act = Activation(config.user_act, config.maxlen)

        self._init_weights(self.ffn)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, q: Tensor, q_mask: Tensor, k: Tensor, k_mask: Tensor) -> Tensor:
        # Attention part
        s = self.attn_layer(q, k, k,
                            q_mask=q_mask, k_mask=k_mask, causal=None)
        if self.config.res_user_block == "mul":
            s *= q
        elif self.config.res_user_block == "sum":
            s += q
        if self.config.user_FF:
            s = self.ffn(s)
        if self.config.user_act:
            s = self.act(s)
        return s  # [B,L,H]


class Activation(nn.Module):
    def __init__(self, act, maxlen=None):
        super().__init__()
        if act == "leakyrelu":
            self.act = nn.LeakyReLU(negative_slope=0.2)
        if act == "swiglu":
            self.act = SwiGLU(embed_dim=maxlen)
        if act == "swiglu2":
            self.act = SwiGLU2()
        if act == "silu":
            self.act = Silu()
        else:
            self.act = IdentityEncoding()

    def forward(self, x):
        return self.act(x)


class SwiGLU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_out = x.clone()
        gate = nn.functional.silu(x.clone())
        return gate * x_out


class SwiGLU(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # x_out, gate = x.chunk(2, dim=-1)
        x_out = self.V(x)
        gate = nn.functional.silu(self.W(x))
        return gate * x_out


class Silu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.silu(x)
