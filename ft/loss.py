from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from treevalue import FastTreeValue, TreeValue, reduce_


class FocalLoss(nn.Module):
    """
    Based on https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
    """

    def __init__(self, num_classes, gamma=2., reduction='mean', weight=None):
        nn.Module.__init__(self)
        self.num_classes = num_classes
        weight = torch.as_tensor(weight).float() if weight is not None else weight
        self.register_buffer('weight', weight)
        self.weight: torch.Tensor

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            labels,
            weight=self.weight,
            reduction=self.reduction
        )


def _tree_sum(x: TreeValue):
    return reduce_(x, lambda **kwargs: torch.sum(torch.stack(list(kwargs.values()))))


class MultiHeadFocalLoss(nn.Module):
    def __init__(self, column_n_classes: Mapping[str, int], columns_weights: Mapping[str, float] = None,
                 gamma=2., reduction='mean', weight=None):
        nn.Module.__init__(self)
        self.loss = {
            name: FocalLoss(n_classes, gamma, reduction, weight)
            for name, n_classes in column_n_classes.items()
        }
        self.loss_tv = FastTreeValue(self.loss)

        columns_weights = dict(columns_weights or {})
        self.column_weights = {
            name: torch.tensor(columns_weights.get(name, 1.0))
            for name in column_n_classes.keys()
        }
        self.column_weights_tv = FastTreeValue(self.column_weights)

    def forward(self, logits, labels):
        return _tree_sum(self.loss_tv(logits, labels) * self.column_weights_tv) / _tree_sum(self.column_weights_tv)
