import torch
import torch.nn as nn


class DeterministicDualEncoderUNet(nn.Module):
    def __init__(self, target_channels=1, dynamic_channels=9, static_channels=2, base_dim=64):
        super().__init__()

        # --- Static Topographical Encoder (Branch A) ---
        self.static_conv1 = nn.Conv2d(static_channels, base_dim // 2, kernel_size=3, padding=1)
        self.static_conv2 = nn.Conv2d(base_dim // 2, base_dim, kernel_size=4, stride=2, padding=1)

        # --- Dynamic Atmospheric Encoder (Branch B) ---
        self.dyn_conv1 = nn.Conv2d(dynamic_channels, base_dim, kernel_size=3, padding=1)
        self.dyn_down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=4, stride=2, padding=1)
        self.dyn_down2 = nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=4, stride=2, padding=1)

        # --- Decoder (Expanding Path) ---
        self.up1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_dim * 5, base_dim, kernel_size=4, stride=2, padding=1)

        final_in_channels = int(base_dim * 2.5)
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_channels, base_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, target_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_dyn, x_stat, extract_features=False):
        # 1. Static Topographical Pass
        s1 = torch.nn.functional.gelu(self.static_conv1(x_stat))
        s2 = torch.nn.functional.gelu(self.static_conv2(s1))

        # 2. Dynamic Atmospheric Pass
        d1 = torch.nn.functional.gelu(self.dyn_conv1(x_dyn))
        d2 = torch.nn.functional.gelu(self.dyn_down1(d1))

        # The bottleneck representation
        d3 = torch.nn.functional.gelu(self.dyn_down2(d2))

        # 3. Decoder Pass
        u1 = torch.nn.functional.gelu(self.up1(d3))

        u1_concat = torch.cat([u1, d2, s2], dim=1)
        u2 = torch.nn.functional.gelu(self.up2(u1_concat))

        u2_concat = torch.cat([u2, d1, s1], dim=1)
        out = self.final_conv(u2_concat)

        # Yield the features for correlation and adversarial alignment
        if extract_features:
            return out, d3
        return out
