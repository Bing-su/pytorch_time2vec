from math import sqrt
from typing import Callable

import torch
from torch import nn


class Time2Vec(nn.Module):
    def __init__(
        self,
        in_features: int,
        embedding_dim: int,
        features_dim: int,
        func: Callable[[torch.Tensor], torch.Tensor] = torch.sin,
    ):
        """
        a Time2Vec layer based on:
        [1] "Time2Vec: Learning a Vector Representation of Time",
             Kazemi et al., 2019.

        Parameters
        ----------
        in_features : int
            number of input features == sequence length
        embedding_dim : int
            number of output features per input column
        features_dim : int
            number of input columns
        func : Callable[[Tensor], Tensor], optional
            Torch periodic function (sin, cos, etc), by default torch.sin

        Return
        ------
        torch.Tensor
            shape of `(batch_size, in_features, (embedding_dim + 1) * features_dim)`
        """
        super().__init__()
        self.in_features = in_features
        self.embedding_dim = embedding_dim
        self.features_dim = features_dim
        self.func = func

        embedding_out = embedding_dim * features_dim

        self.W0 = nn.Parameter(torch.empty(in_features, in_features))
        self.b0 = nn.Parameter(torch.empty(in_features, features_dim))
        self.W = nn.Parameter(torch.empty(features_dim, embedding_out))
        self.b = nn.Parameter(torch.empty(in_features, embedding_out))

        self.reset_parameters()

    def reset_parameters(self):
        "https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear"
        nn.init.kaiming_uniform_(self.W0, a=sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=sqrt(5))

        fan_in0, _ = nn.init._calculate_fan_in_and_fan_out(self.W0)
        bound0 = 1 / sqrt(fan_in0) if fan_in0 > 0 else 0
        nn.init.uniform_(self.b0, -bound0, bound0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # tau : (batch_size, in_features, features_dim)
        batch_size = tau.shape[0]

        W0 = self.W0.repeat(batch_size, 1, 1)
        b0 = self.b0.repeat(batch_size, 1, 1)
        W = self.W.repeat(batch_size, 1, 1)
        b = self.b.repeat(batch_size, 1, 1)

        original = torch.bmm(W0, tau) + b0
        trans = self.func(torch.bmm(tau, W) + b)

        # output : (batch_size, in_features, (embedding_dim + 1) * features_dim)
        output = torch.cat([trans, original], dim=-1)
        return output
