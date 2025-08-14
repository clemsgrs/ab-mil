import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A simple MLP classifier with a tunable number of hidden layers.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.0
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        input_dim: input feature dimension
        hidden_dim: hidden layer dimension
        dropout: dropout
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout=0.0):
        super(Attn_Net, self).__init__()
        self.net = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        attn = self.net(x)
        return attn  # N x 1


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        input_dim: input feature dimension
        hidden_dim: hidden layer dimension
        dropout: dropout
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout=0.0):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]

        self.attention_b = [
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        ]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        attn = a.mul(b)
        attn = self.attention_c(attn)
        return attn  # N x 1
