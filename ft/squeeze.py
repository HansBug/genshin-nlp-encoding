from torch import nn


class LinearSqueezer(nn.Module):
    def forward(self, x):
        return x.reshape(-1)


class LastSqueezer(nn.Module):
    def forward(self, x):
        return x[..., -1, :]


class MeanSqueezer(nn.Module):
    def forward(self, x):
        return x.mean(axis=-2)
