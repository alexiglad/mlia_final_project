""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, class_weights=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        print("class_weights", class_weights)
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device='cuda')
        else:
            self.class_weights = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        # if self.class_weights is not None:
        #     logprobs = logprobs * self.class_weights.unsqueeze(0)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
