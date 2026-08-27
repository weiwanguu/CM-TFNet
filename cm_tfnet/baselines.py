"""Fair baselines: flat 15-D sequence -> backbone -> Softplus bimanual Fz (same I/O as CM-TFNet)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import CMTFNet

# Widths matched to CM-TFNet(d=64,h=64) ≈ 377030 params (2-layer RNN / standard TCN)
CAPACITY_MATCH_WIDTH = {
    "lstm": 148,  # ~377550
    "gru": 169,  # ~376534
    "tcn": 153,  # ~378524
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _pred_head(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, 2),
        nn.Softplus(),
    )


class LSTMBaseline(nn.Module):
    """(B,T,15) -> (B,2)"""

    def __init__(self, d_model: int = 64, hidden: int = 64, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(15, d_model)
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _pred_head(hidden, dropout)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        h = self.input_proj(x)
        out, _ = self.rnn(h)
        y = self.head(out[:, -1, :])
        if return_gate:
            return y, None
        return y


class GRUBaseline(nn.Module):
    """(B,T,15) -> (B,2)"""

    def __init__(self, d_model: int = 64, hidden: int = 64, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(15, d_model)
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _pred_head(hidden, dropout)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        h = self.input_proj(x)
        out, _ = self.rnn(h)
        y = self.head(out[:, -1, :])
        if return_gate:
            return y, None
        return y


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBaseline(nn.Module):
    """Causal dilated TCN with receptive field covering the window; predict from the last step."""

    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(15, d_model)
        layers = []
        for dil in (1, 2, 4, 8, 16):
            layers.append(
                nn.Sequential(
                    CausalConv1d(d_model, d_model, kernel_size=3, dilation=dil),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = _pred_head(d_model, dropout)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        h = self.input_proj(x)
        y = h.transpose(1, 2)
        for blk in self.blocks:
            y = y + blk(y)
        h = self.norm(y.transpose(1, 2))
        out = self.head(h[:, -1, :])
        if return_gate:
            return out, None
        return out


def build_model(
    name: str,
    d_model: int = 64,
    gru_hidden: int = 64,
    dropout: float = 0.1,
    num_layers: int = 2,
    match_capacity: bool = False,
) -> nn.Module:
    from .ablations import ABLATION_ALIASES, build_ablation

    key = name.strip().lower()
    if key in ("cm_tfnet", "cmtfnet", "ours"):
        return CMTFNet(d_model, gru_hidden, dropout)

    if key in ABLATION_ALIASES:
        return build_ablation(key, d_model, gru_hidden, dropout)

    if match_capacity and key in CAPACITY_MATCH_WIDTH:
        w = CAPACITY_MATCH_WIDTH[key]
        d_model = w
        gru_hidden = w

    if key == "lstm":
        return LSTMBaseline(d_model, gru_hidden, dropout, num_layers=num_layers)
    if key == "gru":
        return GRUBaseline(d_model, gru_hidden, dropout, num_layers=num_layers)
    if key == "tcn":
        return TCNBaseline(d_model, dropout)
    raise ValueError(f"unknown model: {name}")
