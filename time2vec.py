from math import sqrt
from typing import Callable

import torch
from torch import nn


class Time2Vec(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        features_dim: int,
        func: Callable[[torch.Tensor], torch.Tensor] = torch.sin,
    ):
        """
        a Time2Vec layer based on:
            "Time2Vec: Learning a Vector Representation of Time",
             Kazemi et al., 2019.

        Parameters
        ----------
        in_features : int
            number of input features. i.e. sequence length
        out_features : int
            number of output features, must be >= 2.
        features_dim : int
            number of input columns.
        func : Callable[[Tensor], Tensor], optional
            Torch periodic function (sin, cos, etc), by default torch.sin

        Returns
        -------
        torch.Tensor
            shape of `(batch_size, out_features, features_dim)`
        """
        super().__init__()

        if out_features < 2:
            raise ValueError("out_features must be >= 2.")

        self.in_features = in_features
        self.out_features = out_features
        self.features_dim = features_dim
        self.func = func

        self.weight = nn.Parameter(torch.empty(features_dim, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features, features_dim))

        self.reset_parameters()

    def reset_parameters(self):
        "https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear"
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: (batch_size, in_features, features_dim)
        x = torch.einsum("foi,bif->bof", self.weight, tau) + self.bias
        out = torch.cat([x[:, :1, :], self.func(x[:, 1:, :])], dim=1)
        # out: (batch_size, out_features, features_dim)
        return out

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, features_dim={}".format(
            self.in_features, self.out_features, self.features_dim
        )
