"""VGG-style CNN baseline for SED — no recurrent or attention layers."""
import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_convs: int = 2, pool=(2, 2)):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_convs):
            layers += [
                nn.Conv2d(ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        layers += [nn.MaxPool2d(pool), nn.Dropout2d(0.1)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG(nn.Module):
    """VGG-style SED model.

    Input : (B, 1, n_mels, T)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 3, n_mels: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(1,   32,  n_convs=2),
            VGGBlock(32,  64,  n_convs=2),
            VGGBlock(64,  128, n_convs=2),
            VGGBlock(128, 128, n_convs=1),
        )
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
    m = VGG(num_classes=3)
    print(f"VGG params: {m.count_parameters():,}")
    x = torch.randn(2, 1, 64, 313)
    print(f"Output: {m(x).shape}")
