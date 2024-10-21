#  # PyTorch
import torch
import torch.nn as nn
import copy

from src.Encoder_Decoder import EncoderBlock, DecoderBlock
from src.Encoding import ItemEncoding
from src.FeedFordward import UserBlock
from src.metrics_and_losses import BPRLoss, BinaryCrossEntropy
from src.utils import get_mask2


class Skeleton(nn.Module):
    def __init__(self, config, Features, device):
        """
        :param config: MyConfig object with attributes for the configuration and training
        :param ItemFeatures: array with shape = (n_items, n_attributes)
        """
        super().__init__()
        self.config = config

        # This encoding has to be applied to seq, neg, pos and test
        self.items_encoding = ItemEncoding(config, Features, device)  # feat_dim)
        self.drop_before_encoding = nn.Dropout(config.hidden_dropout_prob)

        # Encoder blocks
        if not config.add_users:
            # ROPE in only the first MHA
            if config.positional_encoding_type == "rope1":
                config._add(**{
                    "positional_encoding_type": "",
                })
                config2 = copy.deepcopy(config)
                config1 = copy.deepcopy(config)
                config1._add(**{
                    "ROPE_encoder": True,
                })
                if config.num_encoder_blocks > 1:
                    self.encoders = nn.ModuleList([EncoderBlock(config1)] +
                                                  [EncoderBlock(config2) for _ in range(config.num_encoder_blocks-1)])
                else:
                    self.encoders = nn.ModuleList([EncoderBlock(config1)])
            else:
                self.encoders = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_blocks)])
        else:
            self.user_encoder = UserBlock(config)
            num_enc = config.num_encoder_blocks - 1 if config.num_encoder_blocks != 1 else 1
            self.encoders = nn.ModuleList([EncoderBlock(config) for _ in range(num_enc)])

        # Layers to apply at the end
        if config.normalization_type == "ln":
            self.norm = nn.LayerNorm(config.embedding_d, config.layer_norm_eps)
        elif config.normalization_type == "gn":
            self.norm = nn.GroupNorm(config.num_groups, config.embedding_d, config.layer_norm_eps)

        # Decoder blocks
        self.decoders = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_blocks)])
        # self.logit_out = nn.Linear(config.hidden_units, 1)
        # nn.init.xavier_uniform_(self.logit_out.weight)

        self.loss_fct = BPRLoss() if config.loss_type == "BPR" else BinaryCrossEntropy()
        self.loss_type = config.loss_type
        self.add_users = config.add_users
        if self.add_users:
            self.user_embedding = nn.Embedding(config.n_users, config.embedding_d, padding_idx=0)
        self.first, self.first_val, self.showme = True, True, True

    def Encoder(self, seq, seqcxt, user):
        # ### MASK - Encoding
        mask = get_mask2(seq)
        x = self.items_encoding(seq, seqcxt, mask)

        # ### ENCODER BLOCKS
        x = self.drop_before_encoding(x)
        if self.config.add_users:
            x *= mask.unsqueeze(2)
            user *= mask.unsqueeze(2)
            x = self.user_encoder(
                q=user,
                q_mask=mask,
                k=x,
                k_mask=mask,
            )
        x *= mask.unsqueeze(2)
        for idx, encoder_layer in enumerate(self.encoders):
            x = encoder_layer(
                x=x,
                mask=mask,
                user=user,
            )
            x *= mask.unsqueeze(2)
        # Normalization after all the encoding blocks
        x = self.norm(x)  # [B,L,H]
        return x, mask

    def U_Encoder(self, user_seq):
        if self.add_users:
            user_emb = self.user_embedding(user_seq).squeeze(1)
        else:
            user_emb = None
        return user_emb

    def Decoder(self, seq, seqcxt, p, p_mask):
        # Encoding
        o_mask = get_mask2(seq)
        x = self.items_encoding(seq, seqcxt, o_mask)
        for idx, decoder_layer in enumerate(self.decoders):
            x = decoder_layer(
                o=x,
                o_mask=o_mask,
                p=p,
                p_mask=p_mask,
            )
        return x, o_mask

    def forward(self, user_seq, seq, seqcxt, pos=None, neg=None,
                poscxt=None, negcxt=None, test=None, testcxt=None):
        """
        When test is different of None this outputs the logits otherwise it computes the loss and this is the output
        :param test: (B x L) ; L = [n, rand, ..., rand]
        :param user_seq: (B x L) ;
        :param seq: (B x L) ; L = [..., 0, 1, 2,..., n-2, n-1] = GIVEN INFORMATION
        :param pos: (B x L) ; L = [..., 1, 2,..., n-1, n] = TARGET TO RANK
        :param neg: (B x L) ; L = [..., rand, rand] = just a tools
        :param seqcxt, poscxt, negcxt: (B x L x CXT)
        :return: float is test is None (B x 1, B x 1 B x 1)
        """
        user_emb = self.U_Encoder(user_seq)
        p, p_mask = self.Encoder(seq, seqcxt, user_emb)

        # ### DECODER: TEST AND TRAIN
        pos_emb, neg_emb, test_emb = None, None, None
        # TODO: preselect the target items based on contrastive learning
        # TODO: Add user information through the queries
        if neg is not None:
            neg_emb, neg_mask = self.Decoder(neg, negcxt, p, p_mask)
            neg_emb = neg_emb.squeeze()
        if pos is not None:
            pos_emb, pos_mask = self.Decoder(pos, poscxt, p, p_mask)
            pos_emb = pos_emb.squeeze()
        if test is not None:
            test_emb, _ = self.Decoder(test, testcxt, p, p_mask)
            test_emb = test_emb.squeeze()

        # LOSS OR LOGITS
        if test_emb is not None:
            return torch.sigmoid(test_emb)
        else:
            return self.loss(seq, torch.sigmoid(pos_emb), torch.sigmoid(neg_emb), pos_mask, neg_mask)

    def loss(self, seq, pos_emb, neg_emb, pos_mask, neg_mask):
        """
        :param seq, pos_emb, neg_emb, pos_mask, neg_mask: [B, L]
        :return: float
        """
        y_true_pos = torch.where(seq > 0, 1.0, 0.0)
        y_true_neg = torch.zeros(seq.size()).to(seq.get_device())
        y_true = torch.cat((y_true_pos, y_true_neg), dim=-1)
        y_pred = torch.cat((pos_emb, neg_emb), dim=-1)
        loss_mask = torch.cat((pos_mask, neg_mask), dim=-1)
        loss = self.loss_fct.forward(y_pred, y_true, loss_mask)
        return loss
