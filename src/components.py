import torch
import torch.nn as nn


def nll_loss(hazards, survival, Y, c, alpha=0.4, eps=1e-7):
    """
    Continuous time scale divided into k discrete bins: T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
    Y = T_discrete is the discrete event time:
        - Y = -1 if T_cont \in (-inf, 0)
        - Y = 0 if T_cont \in [0, a_1)
        - Y = 1 if T_cont in [a_1, a_2)
        - ...
        - Y = k-1 if T_cont in [a_(k-1), inf)
    hazards = discrete hazards, hazards(t) = P(Y=t | Y>=t, X) for t = -1, 0, 1, 2, ..., k-1
    survival = survival function, survival(t) = P(Y > t | X)

    All patients are alive from (-inf, 0) by definition, so P(Y=-1) = 0
    -> hazards(-1) = 0
    -> survival(-1) = P(Y > -1 | X) = 1

    Summary:
        - neural network is hazard probability function, h(t) for t = 0, 1, 2, ..., k-1
        - h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 0, 1, 2, ..., k-1
    c = c.view(batch_size, 1).float()  # censoring status, 0 or 1
    if survival is None:
        survival = torch.cumprod(
            1 - hazards, dim=1
        )  # survival is cumulative product of 1 - hazards
    survival_padded = torch.cat(
        [torch.ones_like(c), survival], 1
    )  # survival(-1) = 1, all patients are alive from (-inf, 0) by definition
    # after padding, survival(t=-1) = survival[0], survival(t=0) = survival[1], survival(t=1) = survival[2], etc
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(survival_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(
        torch.gather(survival_padded, 1, Y + 1).clamp(min=eps)
    )
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, label, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, label, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, label, c, alpha=alpha)


class MLP(nn.Module):
    """
    A simple MLP classifier with a tunable number of hidden layers.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
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

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout=0.0):
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

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout=0.0):
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
