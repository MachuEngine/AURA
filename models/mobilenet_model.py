"""MobileNetV2-style lightweight CNN for SED.

Uses inverted residual blocks with depthwise separable convolutions.
Adapted for single-channel audio spectrograms.
"""
import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 6):
        super().__init__()
        mid = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.conv = nn.Sequential(
            # Pointwise expand
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            # Pointwise project
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetSED(nn.Module):
    """MobileNetV2-style SED model.

    Input : (B, 1, n_mels, T)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        # fmt: off
        self.features = nn.Sequential(
            # Initial conv: 1 → 32, stride 2
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            # Inverted residuals (in, out, stride, expand)
            InvertedResidual(32, 16, stride=1, expand=1),
            InvertedResidual(16, 24, stride=2, expand=6),
            InvertedResidual(24, 24, stride=1, expand=6),
            InvertedResidual(24, 32, stride=2, expand=6),
            InvertedResidual(32, 32, stride=1, expand=6),
            InvertedResidual(32, 64, stride=2, expand=6),
            InvertedResidual(64, 64, stride=1, expand=6),
            # Final pointwise conv
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
        )
        # fmt: on
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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
        return self.classifier(self.features(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = MobileNetSED(num_classes=3)
    print(f"MobileNet params: {m.count_parameters():,}")
    x = torch.randn(2, 1, 64, 313)
    print(f"Output: {m(x).shape}")
