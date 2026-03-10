import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalDualEncoderUNet(nn.Module):
    """
    Time-conditioned dual-encoder U-Net for atmospheric super resolution.
    Physically decouples static terrestrial boundaries from dynamic synoptic forcing.
    """

    def __init__(self, target_channels=1, dynamic_channels=9, static_channels=2, base_dim=64):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.GELU(),
            nn.Linear(base_dim * 4, base_dim * 4),
        )

        self.static_conv1 = nn.Conv2d(static_channels, base_dim // 2, kernel_size=3, padding=1)
        self.static_conv2 = nn.Conv2d(base_dim // 2, base_dim, kernel_size=4, stride=2, padding=1)
        self.static_conv3 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=4, stride=2, padding=1)

        dyn_in_channels = target_channels + dynamic_channels
        self.dyn_conv1 = nn.Conv2d(dyn_in_channels, base_dim, kernel_size=3, padding=1)
        self.dyn_down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=4, stride=2, padding=1)
        self.dyn_down2 = nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=4, stride=2, padding=1)

        self.time_emb1 = nn.Linear(base_dim * 4, base_dim * 2)
        self.time_emb2 = nn.Linear(base_dim * 4, base_dim * 4)

        self.up1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_dim * 6, base_dim, kernel_size=4, stride=2, padding=1)

        final_in_channels = int(base_dim * 2.5)
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_channels, base_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, target_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_noisy, time, cond, extract_features=False):
        x_dyn = cond[:, :9, :, :]
        x_stat = cond[:, 9:, :, :]

        t = self.time_mlp(time)

        s1 = torch.nn.functional.gelu(self.static_conv1(x_stat))
        s2 = torch.nn.functional.gelu(self.static_conv2(s1))
        s3 = torch.nn.functional.gelu(self.static_conv3(s2))

        d_in = torch.cat([x_noisy, x_dyn], dim=1)
        d1 = torch.nn.functional.gelu(self.dyn_conv1(d_in))

        d2 = self.dyn_down1(d1)
        d2 = d2 + self.time_emb1(t)[:, :, None, None]
        d2 = torch.nn.functional.gelu(d2)

        d3 = self.dyn_down2(d2)
        d3 = d3 + self.time_emb2(t)[:, :, None, None]
        d3 = torch.nn.functional.gelu(d3)

        u1 = torch.nn.functional.gelu(self.up1(d3))

        u1_concat = torch.cat([u1, d2, s3], dim=1)
        u2 = torch.nn.functional.gelu(self.up2(u1_concat))

        u2_concat = torch.cat([u2, d1, s1], dim=1)
        out = self.final_conv(u2_concat)
        if extract_features:
            return out, d3
        return out


class ConsistencyModel(nn.Module):
    def __init__(self, sigma_data=0.5, epsilon=0.002):
        super().__init__()
        self.net = ConditionalDualEncoderUNet()
        self.sigma_data = sigma_data
        self.epsilon = epsilon

    def forward(self, x, t, cond, extract_features=False):
        c_skip = self.sigma_data**2 / ((t - self.epsilon) ** 2 + self.sigma_data**2)
        c_out = (self.sigma_data * (t - self.epsilon)) / torch.sqrt(self.sigma_data**2 + t**2)
        c_in = 1.0 / torch.sqrt(self.sigma_data**2 + t**2)
        c_noise = 0.25 * torch.log(t + 1e-4)

        c_skip = c_skip.view(-1, 1, 1, 1)
        c_out = c_out.view(-1, 1, 1, 1)
        c_in = c_in.view(-1, 1, 1, 1)
        c_noise = c_noise.view(-1)

        F_x = self.net(x * c_in, c_noise, cond)
        
        if extract_features:
            F_x, features = self.net(x * c_in, c_noise, cond, extract_features=True)
            return c_skip * x + c_out * F_x, features
        else:
            F_x = self.net(x * c_in, c_noise, cond)
            return c_skip * x + c_out * F_x
