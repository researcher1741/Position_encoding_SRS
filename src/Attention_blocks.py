#  # PyTorch
import torch
import torch.nn as nn
from torch import Tensor

from src.Encoding import RelativePosition, RotaryPositionalEmbeddings


class AttentionMechanism(nn.Module):
    """ This module is a MultiHeadAttention block. """

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float, intercalate_act: bool):
        super().__init__()

        assert embed_dim % num_heads == 0.0, "Embedding dim must be divisible by number of heads"

        self.d = embed_dim
        self.H = num_heads
        self.Hd = embed_dim // num_heads

        self.WQ = nn.Linear(embed_dim, embed_dim)
        self.WK = nn.Linear(embed_dim, embed_dim)
        self.WV = nn.Linear(embed_dim, embed_dim)

        if intercalate_act:
            self.middle_activation = nn.LeakyReLU(0.2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.Hd]))

        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)

        nn.init.zeros_(self.WQ.bias)
        nn.init.zeros_(self.WK.bias)
        nn.init.zeros_(self.WV.bias)

        self.first = True

    def masking(self, q_mask, k_mask, causal=None):
        # Join masks
        mat1, mat2 = q_mask.unsqueeze(1).transpose(1, 2), k_mask.unsqueeze(1)

        # Mask
        attn_mask = torch.bmm(mat1, mat2).bool()  # batch mul
        attn_mask = torch.tile(attn_mask, (self.H, 1, 1))  # extend two dimensions
        if causal is not None:
            attn_mask = torch.tril(attn_mask, diagonal=causal)
        return attn_mask

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            q_mask: Tensor,
            k_mask: Tensor,
            causal: int = None,
            return_w: bool = False,
    ) -> Tensor:
        pass


class DotHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sig = nn.Sigmoid()

    def forward(self, query: Tensor, key: Tensor, q: Tensor, q_mask: Tensor, k_mask: Tensor, causal=None) -> Tensor:
        y = torch.sum(key * query, dim=-1)  # Dot-product between profile items and target items
        y = self.sig.forward(y)  # Apply sigmoid activation
        y = y * q_mask  # [B, L] in code
        return y


class MHA(AttentionMechanism):
    """ This module is a MultiHeadAttention block. """

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float, intercalate_act: bool):
        super().__init__(embed_dim, num_heads, dropout, intercalate_act)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            q_mask: Tensor,
            k_mask: Tensor,
            causal: int = None,
            return_w: bool = False,
    ) -> Tensor:
        if self.first:
            self.scale = self.scale.to(query.get_device())
            self.first = False

        query = self.WQ(query)  # [B,L,H]
        key = self.WK(key)  # [B,L,H]
        value = self.WV(value)  # [B,L,H]

        if hasattr(self, 'middle_activation'):
            query = self.middle_activation(query)
            key = self.middle_activation(key)
            value = self.middle_activation(value)

        query = torch.cat(torch.split(query, self.Hd, dim=2), dim=0)  # [B,L,H]
        key = torch.cat(torch.split(key, self.Hd, dim=2), dim=0)  # [B,L,H]
        value = torch.cat(torch.split(value, self.Hd, dim=2), dim=0)  # [B,L,H]

        attn_mask = self.masking(q_mask, k_mask, causal)  # bool mask
        add_mask = torch.where(attn_mask, 0.0, -1e10)  # True = 0, False = -1e10

        weights = torch.baddbmm(add_mask, query, key.transpose(1, 2))  # batch mul with add mask
        weights = weights / self.scale
        weights = self.softmax(weights) * attn_mask
        out = self.dropout(weights)

        out = torch.bmm(out, value)
        out = torch.cat(torch.split(out, out.shape[0] // self.H, dim=0), dim=2)

        if return_w:
            return weights, out
        else:
            return out


class RMHA(AttentionMechanism):
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float, intercalate_act: bool, max_relative_position: int):
        super().__init__(embed_dim, num_heads, dropout, intercalate_act)

        self.relative_position_k = RelativePosition(self.Hd, max_relative_position)
        self.relative_position_v = RelativePosition(self.Hd, max_relative_position)

        self.fc_o = nn.Linear(embed_dim, embed_dim)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            q_mask: Tensor,
            k_mask: Tensor,
            causal: int = None,
            return_w: bool = False,
    ) -> Tensor:
        if self.first:
            self.scale = self.scale.to(query.get_device())
            self.first = False

        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)

        if hasattr(self, 'middle_activation'):
            query = self.middle_activation(query)
            key = self.middle_activation(key)
            value = self.middle_activation(value)

        # attn1
        r_q1 = query.view(batch_size, -1, self.H, self.Hd).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.H, self.Hd).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        # attn2
        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.H, self.Hd)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.H, len_q, len_k)
        weights = (attn1 + attn2) / self.scale

        attn_mask = self.masking(q_mask, k_mask, causal)  # bool mask
        if self.H != 1:
            a = torch.split(attn_mask, [batch_size for _ in range(self.H)], dim=0)
            attn_mask = torch.cat(tuple(i.unsqueeze(1) for i in a), dim=1)
            weights = weights + torch.where(attn_mask, 0.0, -1e10)
            weights = self.softmax(weights) * attn_mask
        else:
            weights = weights + torch.where(attn_mask, 0.0, -1e10).unsqueeze(1)
            weights = self.softmax(weights) * attn_mask.unsqueeze(1)
        out = self.dropout(self.softmax(weights))

        # out = [B, num_heads, Qlen, key len]
        r_v1 = value.view(batch_size, -1, self.H, self.Hd).permute(0, 2, 1, 3)
        weight1 = torch.matmul(out, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = out.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.H, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.H, len_q, self.Hd)

        x = weight1 + weight2

        # x = [B, num_heads, Qlen, H]
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [B, Qlen, num_heads, H]
        x = x.view(batch_size, -1, self.d)

        # x = [B, Qlen, H]
        x = self.fc_o(x)
        return x


class ROPEMHA(AttentionMechanism):
    """ This module is a MultiHeadAttention block with ROPE embeddings. """

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float, intercalate_act: bool):
        super().__init__(embed_dim, num_heads, dropout, intercalate_act)
        self.rope_embeddings = RotaryPositionalEmbeddings(dimension=self.Hd)

    def apply_rope(self, x: Tensor):
        original_shape = x.size()
        # Split operation
        chunks = torch.split(x, self.Hd, dim=2)
        # Concatenate operation
        x = torch.cat([chunk.unsqueeze(1) for chunk in chunks], dim=1)
        # ROPE
        x = self.rope_embeddings(x)
        # Recover the original shape
        x = x.view(*original_shape)
        # Recover original tensor values after split
        split_indices = [chunk.size(2) for chunk in chunks]
        recovered_chunks = torch.split(x, split_indices, dim=2)
        # Concatenate the recovered chunks along the split dimension
        x_recovered = torch.cat(recovered_chunks, dim=2)
        return x_recovered

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            q_mask: Tensor,
            k_mask: Tensor,
            causal: int = None,
            return_w: bool = False,
    ) -> Tensor:
        if self.first:
            self.scale = self.scale.to(query.get_device())
            self.first = False

        # Linear projections
        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)

        if hasattr(self, 'middle_activation'):
            query = self.middle_activation(query)
            key = self.middle_activation(key)
            value = self.middle_activation(value)

        query = torch.cat(torch.split(query, self.Hd, dim=2), dim=0)
        key = torch.cat(torch.split(key, self.Hd, dim=2), dim=0)
        value = torch.cat(torch.split(value, self.Hd, dim=2), dim=0)

        # Apply ROPE embeddings to query and key tensors
        query = self.apply_rope(query)
        key = self.apply_rope(key)

        attn_mask = self.masking(q_mask, k_mask, causal)  # bool mask
        add_mask = torch.where(attn_mask, 0.0, -1e10)  # True = 0, False = -1e10

        # Compute attention scores
        weights = torch.baddbmm(add_mask, query, key.transpose(1, 2))  # batch mul with add mask
        weights = weights / self.scale
        weights = self.softmax(weights) * attn_mask
        out = self.dropout(weights)

        # Weighted sum with value tensor
        out = torch.bmm(out, value)
        out = torch.cat(torch.split(out, out.shape[0] // self.H, dim=0), dim=2)

        if return_w:
            return weights, out
        else:
            return out
