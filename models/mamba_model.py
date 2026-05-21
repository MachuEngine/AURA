"""CNN + Mamba (S6 SSM) model for SED — pure PyTorch, CPU/MPS/CUDA compatible.

Architecture: same CNN backbone as CRNN → project → N × MambaBlock → GAP → classifier

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces," arXiv 2312.00752.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sed_model import ConvBlock


class MambaBlock(nn.Module):
    """Mamba selective SSM block.

    Pre-norm + residual. Sequential scan is used (O(L·D·N)) — fast enough
    for the short sequences produced by the CNN backbone (L ≈ 19).

    Args:
        d_model : model dimension
        d_state : SSM state size N
        d_conv  : causal depthwise conv kernel size
        expand  : inner dimension multiplier (d_inner = expand × d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(expand * d_model)
        self.d_inner = d_inner
        dt_rank = max(1, math.ceil(d_model / 16))
        self.dt_rank = dt_rank

        self.norm = nn.LayerNorm(d_model)

        # Input projection — produces x-branch and z-gate
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Causal depthwise conv (grouped so channels don't mix)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # Selective projections: Δ (dt), B, C from the input
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # A matrix: log-space HiPPO initialisation (fixed structure)
        A = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(d_inner, 1)
        )
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip-connection scale
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)                                  # (B, L, 2·d_inner)
        x_in, z = xz.chunk(2, dim=-1)                        # each (B, L, d_inner)

        # Causal depthwise conv — keep only first L outputs
        L = x_in.shape[1]
        x_c = self.conv1d(x_in.transpose(1, 2))[:, :, :L]   # (B, d_inner, L)
        x_c = self.act(x_c.transpose(1, 2))                  # (B, L, d_inner)

        # SSM + gating + output proj
        y = self._ssm(x_c) * self.act(z)
        return self.out_proj(y) + residual

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective state space scan.  x: (B, L, d_inner) → (B, L, d_inner)"""
        B, L, D = x.shape
        N = self.d_state

        A = -torch.exp(self.A_log.float())                    # (d_inner, N)

        x_dbl = self.x_proj(x)                                # (B, L, dt_rank+2N)
        dt_raw, B_ssm, C = x_dbl.split([self.dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))                 # (B, L, d_inner)

        # Zero-Order Hold discretisation
        dA = torch.exp(dt.unsqueeze(-1) * A)                  # (B, L, d_inner, N)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)            # (B, L, d_inner, N)

        # Sequential scan — correct for any device, fast for L ≈ 19
        h = x.new_zeros(B, D, N)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i, :, None]
            ys.append((h * C[:, i, None, :]).sum(dim=-1))     # (B, d_inner)
        y = torch.stack(ys, dim=1)                            # (B, L, d_inner)

        return y + x * self.D                                  # skip connection


class CRNNMamba(nn.Module):
    """CNN backbone + Mamba temporal modelling.

    Identical CNN front-end to CRNN for fair architectural comparison.

    Input : (B, 1, n_mels, T)
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = 3,
        n_mels: int = 64,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 2,
    ):
        super().__init__()
        # Shared CNN backbone (1→16→32→48→64, 4×pool2)
        self.cnn = nn.Sequential(
            ConvBlock(1,  16, pool=(2, 2)),
            ConvBlock(16, 32, pool=(2, 2)),
            ConvBlock(32, 48, pool=(2, 2)),
            ConvBlock(48, 64, pool=(2, 2)),
        )

        cnn_out = 64 * (n_mels // 16)                         # 256
        self.input_proj = nn.Linear(cnn_out, d_model)

        self.mamba = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
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
        for layer in self.mamba:
            x = layer(x)
        x = self.norm_out(x).mean(dim=1)                       # global avg pool
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CRNNMamba(num_classes=3)
    print(f"CNN-Mamba params: {m.count_parameters():,}")
    x = torch.randn(2, 1, 64, 313)
    out = m(x)
    print(f"Output: {out.shape}")
