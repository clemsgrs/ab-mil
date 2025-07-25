import torch
import torch.nn as nn

from src.models.utils import MLP, Attn_Net, Attn_Net_Gated


class ABMIL(nn.Module):
    def __init__(
        self,
        features_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        gated: bool = False,
    ):
        super().__init__()
        self.mlp = MLP(
            input_dim=features_dim,
            hidden_dim=hidden_dim,
            output_dim=features_dim,
            num_layers=3,
        )

        if gated:
            self.attention = Attn_Net_Gated(
                input_dim=features_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.attention = Attn_Net(
                input_dim=features_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        self.classifier = nn.Linear(features_dim, num_classes)

    def forward(self, x, return_attn: bool = False, attn_only: bool = False):
        # B: batch size
        # N: number of tiles in input whole-slide
        # D: tile feature dimension
        # K: number of attention heads (K = 1 for AB-MIL)
        # x is B x N x D
        x = self.mlp(x)
        raw_attn = self.attention(x)  # B x N x K
        raw_attn = torch.transpose(raw_attn, -2, -1)  # B x K x N
        attn = raw_attn.softmax(dim=-1)  # softmax over N
        if attn_only:
            return attn
        x = torch.bmm(attn, x).squeeze(dim=1)  # B x K x C --> B x C
        x = self.classifier(x)
        if return_attn:
            return x, attn
        else:
            return x
