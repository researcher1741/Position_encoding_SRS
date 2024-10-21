#  # PyTorch
import torch
import torch.nn as nn
from torch import Tensor

from src.Attention_blocks import MHA, RMHA, DotHead, ROPEMHA
from src.FeedFordward import PointWiseFeedForward, PointWiseFeedForwardOut


class EncoderBlock(nn.Module):
    """
    1. Layer Norm for the queries
    2. Multi-head Attention
    3. Layer Norm
    4. FF
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Architecture
        if config.RMHA_encoder:
            self.attn_layer = RMHA(config.embedding_d,
                                   config.num_heads,
                                   config.attention_probs_dropout_prob,
                                   config.intercalate_act,
                                   config.max_relative_position)
        elif config.ROPE_encoder:
            self.attn_layer = ROPEMHA(config.embedding_d,
                                      config.num_heads,
                                      config.attention_probs_dropout_prob,
                                      config.intercalate_act)
        else:
            self.attn_layer = MHA(config.embedding_d,
                                  config.num_heads,
                                  config.attention_probs_dropout_prob,
                                  config.intercalate_act)
        if config.ln_in_Q_encoder:
            self.norm_on_Q = torch.nn.LayerNorm(config.hidden_units, eps=config.layer_norm_eps)
        if config.ln_in_AH_encoder:
            self.norm_on_top = torch.nn.LayerNorm(config.hidden_units, eps=config.layer_norm_eps)
        if config.dropout_btn_MHA_FF:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # FF block
        self.FFEncoder = PointWiseFeedForward(config)
        # self.FFEncoder = FFBlock(config)

    def forward(self, x: Tensor, mask: Tensor, user=None) -> Tensor:
        # Attention part
        q = self.norm_on_Q(x) if self.config.ln_in_Q_encoder else x.clone()
        s = self.attn_layer(q, x, x, q_mask=mask, k_mask=mask, causal=None)

        if self.config.residual_connection_encoder == "mul":  # True
            s *= q
        elif self.config.residual_connection_encoder == "sum":  # False
            s += q
        if self.config.ln_in_AH_encoder:  # True
            s = self.norm_on_top(s)
        if self.config.dropout_btn_MHA_FF:  # False
            s = self.dropout(s)

        # MLP part
        if self.config.residual_connection_encoder_FF == "mul":  # "mul" in paper
            s *= self.FFEncoder(s)
        if self.config.residual_connection_encoder_FF == "sum":  # "sum" in code
            s += self.FFEncoder(s)
        else:
            s = self.FFEncoder(s)

        return s  # [B,L,H]


class DecoderBlock(nn.Module):
    """
    1. Multihead attention (decoder, this is without normalization of the queries)
    2. Mask or not
    3. Linear
    4. Reshape
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Attention heads
        if config.decoder_head == "RMHA":
            self.attn_layer = RMHA(config.embedding_d,
                                   config.num_heads,
                                   config.attention_probs_dropout_prob,
                                   config.intercalate_act,
                                   config.max_relative_position)
        elif config.decoder_head == "masked":
            self.attn_layer = MHA(config.embedding_d,
                                  config.num_heads,
                                  config.attention_probs_dropout_prob,
                                  config.intercalate_act)
        elif config.decoder_head == "dot":
            self.attn_layer = DotHead(config.embedding_d,
                                      config.num_heads,
                                      config.attention_probs_dropout_prob,
                                      config.intercalate_act)
        else:
            print("There was no decoding head, or at least not accepted")

        if config.dropout_btn_MHA_FF:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.FFOut = PointWiseFeedForwardOut(config)

    def forward(self, o: Tensor, o_mask: Tensor, p: Tensor, p_mask: Tensor) -> Tensor:
        # causal = -1 if self.training else None
        # Attention part
        s = self.attn_layer(o, p, p, q_mask=o_mask, k_mask=p_mask, causal=None)
        if self.config.residual_connection_decoder == "mul":
            s *= o
        elif self.config.residual_connection_decoder == "sum":
            s += o
        if self.config.dropout_btn_MHA_FF:
            s = self.dropout(s)
        if self.config.mask_before_FF_decoder:
            if self.config.decoder_head != "dot":
                s *= o_mask.unsqueeze(2)  # [B, L, H] in code
            else:
                s *= o_mask  # [B, L, H] in code
        if self.config.decoder_head != "dot":
            s = self.FFOut(s)
        if not self.config.mask_before_FF_decoder:
            s *= o_mask  # [B, L] in code
        return s  # [B,L,H]
