import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import embedding as lookup
import math


class ItemEncoding(nn.Module):
    """
    This block corresponds to the encoding from CARCA, for the item keys and the item features
    """

    def __init__(self, config, Features, device="cuda"):  # , feat_dim):
        super().__init__()
        self.config = config

        self.item_embedding = nn.Embedding(config.n_items + 1,
                                           config.embedding_d,
                                           padding_idx=0,
                                           max_norm=config.max_norm,
                                           norm_type=config.norm_type)  # (57290, 90)
        self.Features = torch.tensor(Features).to(device)

        if config.positional_encoding_type == "absolute":
            self.position_encod = SinPositionalEncoding(config)
        elif config.positional_encoding_type == "rotatory":
            self.position_encod = RotaryEncoding(config)
        elif config.positional_encoding_type == "learnt":
            self.position_encod = LearnedPositionalEncoding(config)
        else:
            self.position_encod = IdentityEncoding()

        self.cxt_att_encoding = nn.Linear(Features.shape[-1] + config.cxt_size, config.embedding_g)
        self.linear_concat2 = nn.Linear(config.embedding_g + config.embedding_d, config.embedding_d)

        self._init_weights(self.cxt_att_encoding)
        self._init_weights(self.linear_concat2)
        # self._init_weights(self.item_embedding)

        if config.PN:
            self.PN = torch.nn.LayerNorm(config.hidden_units, eps=config.layer_norm_eps)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            # torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            # torch.nn.init.xavier_uniform_(module.weight)
            module._fill_padding_idx_with_zero()

    def forward(self, ids: Tensor, cxt: Tensor, mask: Tensor) -> Tensor:
        """
        :param ids: [B, L]
        :param cxt: [B, L, CXT_SIZE], [128,10,6]
        :param mask: [B, L]
        :return:
        """
        # TODO: add the concatenation of the user
        # Extract the attributes
        att = lookup(ids, self.Features)
        # Concat CXT and ATT
        cxt_att = torch.cat([cxt, att], dim=-1)  # [B,L, CTX + ATT]
        features = self.cxt_att_encoding(cxt_att.float())  # [B,L, g]
        ids_emb = self.item_embedding(ids)  # [B,L, d]
        ids_emb *= self.item_embedding.embedding_dim ** 0.5
        x = torch.cat([ids_emb, features], dim=-1)  # [B,L, CTX + H]
        x = self.linear_concat2(x)  # [B,L, H]
        x = self.position_encod(x)  # [B,L, H]
        x *= mask.unsqueeze(2)
        x = self.PN(x) if self.config.PN else x
        return x  # # [B,L,H]


class IdentityEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class SinPositionalEncoding(nn.Module):

    def __init__(self, config):
        """
        This class builds the different options for encoding
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        self.concat = config.position_concatenation
        L, H = config.maxlen, config.embedding_d
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(L, H)  # [L,H] = [75, 90]
        # We compute the exponential for sin and cosine. We use half of the length: [0,1,2,3, ..., L-1]
        position = torch.arange(0, L).unsqueeze(1)  # [L,1] dim for [0,1,2,3, ..., L-1]
        Normalizer = -math.log(10000) / H  #
        div_term = torch.exp(torch.arange(0, H, 2).float() * Normalizer)  # e^2j * Normalizer
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = [L,H] = [75, 90]
        if config.reverse:
            pe = torch.flip(pe, [0])
        # [1, L, H]
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)
        if self.concat:
            self.encoding = nn.Linear(H * 2, H)

    def forward(self, x: Tensor) -> Tensor:
        if not self.concat:
            x = x + self.pe[:, : x.size(1), :]
        else:
            s = x.shape
            x = torch.cat([x, self.pe.expand(s[0], s[1], s[2])], -1)
            x = self.encoding(x)
        return x


class RelativePosition(nn.Module):
    """
    Module for generating relative positional embeddings.

    This module computes relative positional embeddings for sequences of given lengths.
    It utilizes a learnable embeddings table that is initialized with Xavier uniform initialization.

    Args:
    - num_units (int): The number of embedding units for each position.
    - max_relative_position (int): The maximum relative position allowed.

    Attributes:
    - num_units (int): The number of embedding units for each position.
    - max_relative_position (int): The maximum relative position allowed.
    - embeddings_table (nn.Parameter): Learnable parameter representing the embeddings table.

    Methods:
    - forward(length_q, length_k): Compute relative positional embeddings for given sequence lengths.

    Example:
    >> relative_position = RelativePosition(num_units=512, max_relative_position=128)
    >> embeddings = relative_position(10, 12)
    """

    def __init__(self, num_units, max_relative_position):
        """
        Initialize the RelativePosition module.

        Args:
        - num_units (int): The number of embedding units for each position.
        - max_relative_position (int): The maximum relative position allowed.

        """
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        """
        Compute relative positional embeddings for given sequence lengths.

        Args:
        - length_q (int): Length of the query sequence.
        - length_k (int): Length of the key sequence.

        Returns:
        torch.Tensor: Relative positional embeddings for the given lengths.

        """
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()
        return embeddings


class LearnedPositionalEncoding(nn.Module):
    """
    Module for applying learned positional encoding to input sequences.

    This module incorporates learnable positional embeddings into input sequences.
    It supports two modes: concatenation and addition. In concatenation mode, the
    positional embeddings are concatenated with the input, followed by a linear transformation.
    In addition mode, the positional embeddings are added directly to the input.

    Args:
    - config (object): Configuration object with the following attributes:
        - position_concatenation (bool): Whether to concatenate positional embeddings with input.
        - maxlen (int): Maximum length of input sequences.
        - embedding_d (int): Dimensionality of the input embeddings.

    Attributes:
    - concat (bool): Whether to concatenate positional embeddings with input.
    - position_embeddings (nn.Embedding): Embedding layer for positional embeddings.
    - encoding (nn.Linear): Linear layer for concatenation mode.

    Methods:
    - forward(x: Tensor) -> Tensor: Apply learned positional encoding to input tensor.

    Example:
    >> config = Configuration(position_concatenation=True, maxlen=100, embedding_d=512)
    >> learned_encoder = LearnedPositionalEncoding(config)
    >> input_tensor = torch.rand((batch_size, sequence_length, embedding_dim))
    >> output_tensor = learned_encoder(input_tensor)

    """

    def __init__(self, config):
        """
        Initialize the LearnedPositionalEncoding module.

        Args:
        - config (object): Configuration object with required attributes.

        """
        super().__init__()
        self.concat = config.position_concatenation
        L, H = config.maxlen, config.embedding_d

        # Learnable positional embeddings
        self.position_embeddings = nn.Embedding(L, H)

        if self.concat:
            self.encoding = nn.Linear(H * 2, H)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply learned positional encoding to the input tensor.

        Args:
        - x (Tensor): Input tensor with shape (batch_size, sequence_length, embedding_dim).

        Returns:
        Tensor: Output tensor after applying learned positional encoding.

        """
        position_ids = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        position_embeddings = self.position_embeddings(position_ids)

        if not self.concat:
            x = x + position_embeddings
        else:
            x = torch.cat([x, position_embeddings], -1)
            x = self.encoding(x)
        return x


class RotaryEncoding(nn.Module):
    """
    Module for applying rotary positional encoding to input sequences.

    This module incorporates positional information into input sequences using rotary positional encoding.
    It supports two modes: concatenation and addition. In concatenation mode, the positional embeddings
    are concatenated with the input, followed by a linear transformation. In addition mode, the positional
    embeddings are added directly to the input.

    Args:
    - config (object): Configuration object with the following attributes:
        - position_concatenation (bool): Whether to concatenate positional embeddings with input.
        - maxlen (int): Maximum length of input sequences.
        - embedding_d (int): Dimensionality of the input embeddings.

    Attributes:
    - concat (bool): Whether to concatenate positional embeddings with input.
    - position_embeddings (nn.Embedding): Embedding layer for positional embeddings.
    - encoding (nn.Linear): Linear layer for concatenation mode.

    Methods:
    - forward(x: Tensor) -> Tensor: Apply rotary positional encoding to input tensor.

    Example:
    >> config = Configuration(position_concatenation=True, maxlen=100, embedding_d=512)
    >> rotary_encoder = RotaryPositionalEncoding(config)
    >> input_tensor = torch.rand((batch_size, sequence_length, embedding_dim))
    >> output_tensor = rotary_encoder(input_tensor)
    """

    def __init__(self, config):
        """
        Initialize the RotaryPositionalEncoding module.
        Args:
        - config (object): Configuration object with required attributes.

        """
        super().__init__()
        self.concat = config.position_concatenation
        L, H = config.maxlen, config.embedding_d
        self.position_embeddings = nn.Embedding(L, H)
        if self.concat:
            self.encoding = nn.Linear(H * 2, H)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply rotary positional encoding to the input tensor.

        Args:
        - x (Tensor): Input tensor with shape (batch_size, sequence_length, embedding_dim).

        Returns:
        Tensor: Output tensor after applying rotary positional encoding.

        """
        # position_ids => L x H, rows [ 0, 1, 2, ...,H]
        position_ids = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        position_embeddings = self.position_embeddings(position_ids)

        # Rotary Positional Encoding
        angles = position_embeddings / 10000.0
        angle_rads = angles[:, :, 0::2] * 2 * math.pi

        sin_angles = torch.sin(angle_rads)
        cos_angles = torch.cos(angle_rads)

        # Add rotation
        sin_angles = sin_angles * torch.tensor([(-1) ** i for i in range(sin_angles.size(-1))], device=x.device)

        # Combine sine and cosine embeddings
        position_embeddings[:, :, 0::2] = sin_angles
        position_embeddings[:, :, 1::2] = cos_angles

        if not self.concat:
            x = x + position_embeddings
        else:
            x = torch.cat([x, position_embeddings], -1)
            x = self.encoding(x)
        return x


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dimension, base: int = 1_000):
        super().__init__()
        self.base = base
        self.dimension = dimension
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        seq_len = x.shape[0]
        theta = 1. / (self.base ** (torch.arange(0, self.dimension, 2).float() / self.dimension)).to(x.device)
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        # Dimensions: [L, embed_dim/2] -> [L, 2 * (embed_dim/2)]
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]
        print(self.cos_cached.size)

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.dimension // 2
        # Dimensions: [B, num_heads, L, embed_dim]
        # -> [B, num_heads, L, embed_dim/2] + [B, num_heads, L, embed_dim/2]
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        # Dimensions: [B, num_heads, L, embed_dim]
        # -> [B, num_heads, L, embed_dim], [B, num_heads, L, other_dims]
        x_rope, x_pass = x[..., :self.dimension], x[..., self.dimension:]
        neg_half_x = self._neg_half(x_rope)
        # Dimensions: [B, num_heads, L, embed_dim], [B, num_heads, L, embed_dim/2]
        # -> [B, num_heads, L, embed_dim]
        positive_part = (x_rope * self.cos_cached[:x.shape[0]])
        negative_part = (neg_half_x * self.sin_cached[:x.shape[0]])
        x_rope = positive_part + negative_part
        # Dimensions: [B, num_heads, L, embed_dim], [B, num_heads, L, other_dims]
        # -> [B, num_heads, L, embed_dim + other_dims]
        return torch.cat((x_rope, x_pass), dim=-1)
