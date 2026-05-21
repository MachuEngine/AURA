"""CNN + Transformer (AST-style) model for SED.

Identical CNN front-end to CRNN; replaces Bi-GRU with a
multi-head self-attention Transformer encoder.  Serves as the
large-model reference upper bound in the ablation study.

Input : (B, 1, n_mels, T)
Output: (B, num_classes) logits
"""
import math

import torch
import torch.nn as nn

from models.sed_model import ConvBlock


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        return self.dropout(x + self.pe[:, : x.size(1)])


class CRNNTransformer(nn.Module):
    """CNN backbone + Transformer encoder (AST-style).

    Identical CNN front-end to CRNN for fair architectural comparison.

    Input : (B, 1, n_mels, T)
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = 3,
        n_mels: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Shared CNN backbone
        self.cnn = nn.Sequential(
            ConvBlock(1,  16, pool=(2, 2)),
            ConvBlock(16, 32, pool=(2, 2)),
            ConvBlock(32, 48, pool=(2, 2)),
            ConvBlock(48, 64, pool=(2, 2)),
        )

        cnn_out = 64 * (n_mels // 16)                         # 256
        self.input_proj = nn.Linear(cnn_out, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,                                   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                                        # (B, 64, n_mels//16, T//16)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)       # (B, T, 256)
        x = self.input_proj(x)                                 # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)                               # (B, T, d_model)
        x = x.mean(dim=1)                                      # global avg pool
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CRNNTransformer(num_classes=3)
    print(f"CNN-Transformer params: {m.count_parameters():,}")
    x = torch.randn(2, 1, 64, 313)
    print(f"Output: {m(x).shape}")
