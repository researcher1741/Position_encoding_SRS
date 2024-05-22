"""
###########    EXAMPLE    ###########
metrics_calculator = SRS_Metrics(device="cpu", myformat=np.array, reverse=True, k=4)
y_true = torch.tensor([[1, 2, 1], [1, 2, 1]])
y_score = torch.tensor([[   [0.8700, 0.2900, 0.1000, 0.0223, 0.0223, 0.0223],
                            [0.2700, 0.1900, 0.0100,0.1700, 0.1200, 0.2100],
                            [0.3700, 0.4200, 0.1220, 0.2700, 0.1900,  0.1000]],
                           [[0.8700, 0.2900, 0.1000, 0.0223, 0.0223, 0.0223],
                            [0.2700, 0.1900, 0.0100,0.1700, 0.1200, 0.2100],
                            [0.3700, 0.2600, 0.1220, 0.2700, 0.1900,  0.1000]]])
print(y_true.size(), y_score.size())
NDCG, HIT = metrics_calculator.NDCG_HIT(true=y_true, score=y_score)  # [B x L], [B x L x V]
NDCG, HIT
# >>> (array(0.75, dtype=float32), array(1., dtype=float32))
"""
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor


class SRS_Metrics:
    def __init__(self, device, myformat, L, reverse=True, k=10):
        self.target_position = -1 if reverse else 0
        self.device = device
        self.myformat = myformat  # np.array or torch.tensor
        self.k = k
        self.L = L

    def check(self, true, score):
        """
        Returns the tensors and the number of dimensions.
        score: B x "L" x "V"
        true: B x "L"
        """
        # Data type correction
        if torch.is_tensor(score) and torch.is_tensor(true):
            pass
        elif isinstance(score, (np.ndarray, np.generic)) and isinstance(true, (np.ndarray, np.generic)):
            true = torch.tensor(true)
            score = torch.tensor(score)
        elif isinstance(true, (np.ndarray, np.generic)):
            true = torch.tensor(true)
        elif isinstance(score, (np.ndarray, np.generic)):
            score = torch.tensor(score)

        # Device correction
        if true.get_device() != self.device:  true = true.to(self.device)
        if score.get_device() != self.device:  score = score.to(self.device)

        # Dimension correction: take the last item from L
        if score.size()[1] == self.L and len(score.size()) == 3:  # B x L x V
            score = score[:, self.target_position, :]
        if score.size()[1] == self.L and len(score.size()) == 2:  # B x L
            score = score[:, self.target_position]
        if true.size()[1] == self.L and len(score.size()) == 2:
            true = true[:, self.target_position]

        # Top 
        _, indices = torch.topk(score, self.k)
        return score, true, indices

    def HIT(self, true, score, indices=None):
        """
        Returns the Hit Ratio on the top k for numpy arrays and torch tensors.
        score: B x V or B x L x V
        true: B or B x L
        """
        if indices is None:
            score, true, indices = self.check(true, score)
        return (indices == true.reshape(-1, 1)).any(1).float().mean()

    def NDCG(self, true, score, indices=None):
        """
        Returns the Normalized Discount Cumulative Gain (NDCG) on the top k for numpy arrays and torch tensors.
        score: B x V or B x L x V
        true: B or B x L
        """
        if indices is None:
            score, true, indices = self.check(true, score)
        matches_matrix = (indices == true.reshape(-1, 1)).float()
        idcg_normalizator = torch.div(1, torch.log2(torch.arange(2, 2 + matches_matrix.shape[-1]))).to(self.device)
        return torch.mul(matches_matrix, idcg_normalizator).sum(-1).mean()

    def NDCG_HIT(self, true, score, indices=None):
        """ Both previous metrics together """
        if indices is None:
            score, true, indices = self.check(true, score)
        HIT = self.HIT(true, score, indices)
        NDCG = self.NDCG(true, score, indices)
        if self.myformat == np.array:
            HIT = HIT.cpu().detach().numpy()
            NDCG = NDCG.cpu().detach().numpy()
        return NDCG, HIT


class BinaryCrossEntropy(nn.Module):
    """ BinaryCrossEntropy, based on Binary Cross entropy
    Args:
        - eps(float): Small value to avoid division by zero
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
        loss = -(y_true * torch.log(y_pred + eps) + (1.0 - y_true) * torch.log(1.0 - y_pred + eps))
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss


class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking
    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.
    """
    def __init__(self, gamma=1e-14):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss
