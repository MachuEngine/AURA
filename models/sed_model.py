"""Lightweight CRNN model for Sound Event Detection (SED).

Architecture: 2D CNN (4 layers) + Bi-GRU + Classification head
Target: <1M parameters, suitable for embedded deployment.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3, 3), pool=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    """Compact CRNN for in-vehicle sound event detection.

    Input: (batch, 1, n_mels, time_frames) log-mel spectrogram
    Output: (batch, num_classes) logits
    """

    def __init__(self, num_classes=4, n_mels=64):
        super().__init__()
        # CNN backbone: 4 conv blocks, channel progression 1→16→32→48→64
        self.cnn = nn.Sequential(
            ConvBlock(1, 16, pool=(2, 2)),
            ConvBlock(16, 32, pool=(2, 2)),
            ConvBlock(32, 48, pool=(2, 2)),
            ConvBlock(48, 64, pool=(2, 2)),
        )

        # Compute GRU input size: n_mels=64 → after 4x pool of 2 → 64/16=4
        gru_input_size = 64 * (n_mels // 16)

        # Bi-GRU temporal modelling
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = self.cnn(x)                          # (B, 64, n_mels//16, T//16)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)               # (B, T, C, F)
        x = x.reshape(B, T, C * F)              # (B, T, C*F)
        x, _ = self.gru(x)                       # (B, T, 128)
        x = x.mean(dim=1)                        # global temporal pooling
        return self.classifier(x)                # (B, num_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CRNN(num_classes=4, n_mels=64)
    params = model.count_parameters()
    print(f"Total trainable parameters: {params:,}")
    assert params < 1_000_000, f"Model too large: {params:,} params"

    x = torch.randn(4, 1, 64, 128)
    out = model(x)
    print(f"Input shape: {x.shape} → Output shape: {out.shape}")
