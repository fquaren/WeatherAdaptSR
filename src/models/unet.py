import torch
import torch.nn as nn
import torch.nn.init as init


def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            init.zeros_(m.bias)


class HeteroscedasticUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder (upsample + conv)
        self.up3 = self.up_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)

        self.up2 = self.up_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)

        self.up1 = self.up_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        # Output is now 2x the number of target channels to predict both the mean and log-variance
        self.output_head = nn.Conv2d(64, out_channels * 2, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Two 3x3 convolutions with ReLU and reflect padding."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_channels, out_channels):
        """Bilinear upsampling followed by a 3x3 convolution."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Ensure correct input size
        assert x.shape[2:] == (128, 128), f"Input must be (128, 128), got {x.shape[2:]}"

        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        # Final output head for both mean and log-variance
        # The output tensor has shape (batch_size, 4, height, width) for out_channels=2
        combined_output = self.output_head(d1)

        # Split the output tensor into two halves for each task
        # The first two channels are for temperature (T), the next two are for precipitation (P)
        pred_T, log_b_T = combined_output[:, 0:1, :, :], combined_output[:, 1:2, :, :]
        pred_P, log_b_P = combined_output[:, 2:3, :, :], combined_output[:, 3:4, :, :]

        return pred_T, log_b_T, pred_P, log_b_P


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder (upsample + conv)
        self.up3 = self.up_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)

        self.up2 = self.up_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)

        self.up1 = self.up_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # Final output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Two 3x3 convolutions with ReLU and reflect padding."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_channels, out_channels):
        """Bilinear upsampling followed by a 3x3 convolution."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Ensure correct input size
        assert x.shape[2:] == (128, 128), f"Input must be (128, 128), got {x.shape[2:]}"

        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output(d1)


# ---------------------------------------------------------- Batch Normalization
class UNet_BN(nn.Module):
    def __init__(self):
        super(UNet_BN, self).__init__()

        # Define encoding layers
        self.encoder1 = self.conv_block(2, 64)  # 1 variable + 1 elevation channel
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, variable, elevation):
        # Check input resolution
        assert (
            variable.shape[2:] == elevation.shape[2:] == (128, 128)
        ), f"Inputs must be same shape (128x128), got {variable.shape[2:]} and {elevation.shape[2:]}"

        # Concatenate input channels
        x = torch.cat((variable, elevation), dim=1)  # Shape: [B, 2, 128, 128]

        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output(d1)


# ---------------------------------------------------------- Dropout
class UNet_DO(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(UNet_DO, self).__init__()

        # Dropout probability
        self.dropout_prob = dropout_prob

        # Define encoding layers
        self.encoder1 = self.conv_block(2, 64)  # 1 variable + 1 elevation channel
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256, dropout_prob=dropout_prob)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, dropout_prob=dropout_prob)

        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256, dropout_prob=dropout_prob)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_prob=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
        )

    def forward(self, variable, elevation):
        # Check input resolution
        assert (
            variable.shape[2:] == elevation.shape[2:] == (128, 128)
        ), f"Inputs must be same shape (128x128), got {variable.shape[2:]} and {elevation.shape[2:]}"

        # Concatenate input channels
        x = torch.cat((variable, elevation), dim=1)  # Shape: [B, 2, 128, 128]

        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output(d1)


# ---------------------------------------------------------- Dropout and Batch Normalization
class UNet_DO_BN(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(UNet_DO_BN, self).__init__()

        # Dropout probability
        self.dropout_prob = dropout_prob

        # Define encoding layers
        self.encoder1 = self.conv_block(2, 64)  # 1 variable + 1 elevation channel
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256, dropout_prob=dropout_prob)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, dropout_prob=dropout_prob)

        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256, dropout_prob=dropout_prob)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_prob=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
        )

    def forward(self, variable, elevation):
        # Check input resolution
        assert (
            variable.shape[2:] == elevation.shape[2:] == (128, 128)
        ), f"Inputs must be same shape (128x128), got {variable.shape[2:]} and {elevation.shape[2:]}"

        # Concatenate input channels
        x = torch.cat((variable, elevation), dim=1)  # Shape: [B, 2, 128, 128]

        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output(d1)


# ---------------------------------------------------------- Noise
class UNet_Noise(nn.Module):
    def __init__(self, noise_std=0.1):
        super(UNet_Noise, self).__init__()

        # Fixed noise standard deviation (non-trainable)
        self.noise_std = noise_std

        # Define encoding layers
        self.encoder1 = self.conv_block(2, 64)  # 1 variable + 1 elevation channel
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, variable, elevation):
        # Apply multiplicative Gaussian noise in training mode
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(variable) * self.noise_std  # Sample noise
            variable_noisy = variable * (1 + noise)  # Multiplicative noise
        else:
            variable_noisy = variable

        # Check input resolution
        assert (
            variable_noisy.shape[2:] == elevation.shape[2:] == (128, 128)
        ), f"Inputs must be same shape (128x128), got {variable_noisy.shape[2:]} and {elevation.shape[2:]}"

        # Concatenate input channels
        x = torch.cat((variable_noisy, elevation), dim=1)  # Shape: [B, 2, 128, 128]

        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output(d1)


# ---------------------------------------------------------- Noise, Dropout and Batch Normalization
class UNet_Noise_DO_BN(nn.Module):
    def __init__(self, dropout_prob=0.3, noise_std=0.1):
        super(UNet_Noise_DO_BN, self).__init__()

        # Dropout probability
        self.dropout_prob = dropout_prob

        # Fixed noise standard deviation (non-trainable)
        self.noise_std = noise_std

        # Define encoding layers
        self.encoder1 = self.conv_block(2, 64)  # 1 variable + 1 elevation channel
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256, dropout_prob=dropout_prob)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, dropout_prob=dropout_prob)

        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256, dropout_prob=dropout_prob)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_prob=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
        )

    def forward(self, variable, elevation):
        # Apply multiplicative Gaussian noise in training mode
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(variable) * self.noise_std  # Sample noise
            variable_noisy = variable * (1 + noise)  # Multiplicative noise
        else:
            variable_noisy = variable

        # Check input resolution
        assert (
            variable_noisy.shape[2:] == elevation.shape[2:] == (128, 128)
        ), f"Inputs must be same shape (128x128), got {variable_noisy.shape[2:]} and {elevation.shape[2:]}"

        # Concatenate input channels
        x = torch.cat((variable_noisy, elevation), dim=1)  # Shape: [B, 2, 128, 128]

        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output(d1)


# ---------------------------------------------------------- Trainable noise
class UNet_Trainable_Noise(nn.Module):
    def __init__(self):
        super(UNet_Trainable_Noise, self).__init__()

        # Input noise parameters (learned)
        self.input_log_var = nn.Parameter(
            torch.zeros(1)
        )  # log variance for input noise

        # Define encoding layers
        self.encoder1 = self.conv_block(2, 64)  # 1 variable + 1 elevation channel
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, variable, elevation):
        # Apply multiplicative Gaussian noise in training mode
        if self.training:
            input_std = torch.exp(
                0.5 * self.input_log_var
            )  # Compute std from log variance
            noise = torch.randn_like(variable) * input_std  # Sample noise
            variable_noisy = variable * (1 + noise)  # Multiplicative noise
        else:
            variable_noisy = variable

        # Check input resolution
        assert (
            variable_noisy.shape[2:] == elevation.shape[2:] == (128, 128)
        ), f"Inputs must be same shape (128x128), got {variable_noisy.shape[2:]} and {elevation.shape[2:]}"

        # Concatenate input channels
        x = torch.cat((variable_noisy, elevation), dim=1)  # Shape: [B, 2, 128, 128]

        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output(d1)


# ---------------------------------------------------------- MMD


def compute_mmd(source_features, target_features, kernel="rbf", sigma=None):
    """
    Computes the Maximum Mean Discrepancy (MMD) between source and target features.

    Args:
        source_features: Tensor of shape (N, D) from the source domain.
        target_features: Tensor of shape (M, D) from the target domain.
        kernel: Kernel type ('rbf' or 'linear').
        sigma: Bandwidth for the RBF kernel (if None, it is estimated using median heuristic).

    Returns:
        MMD loss (scalar)
    """

    def rbf_kernel(X, Y, sigma):
        XX = torch.sum(X**2, dim=1, keepdim=True)  # (N, 1)
        YY = torch.sum(Y**2, dim=1, keepdim=True)  # (M, 1)
        XY = torch.matmul(X, Y.t())  # (N, M)
        dists = torch.clamp(XX - 2 * XY + YY.t(), min=0.0)  # Squared L2 distance
        return torch.exp(-dists / (2 * sigma**2))  # RBF kernel

    # Normalize features to prevent large magnitude issues
    source_features = source_features / (
        source_features.norm(dim=1, keepdim=True) + 1e-6
    )
    target_features = target_features / (
        target_features.norm(dim=1, keepdim=True) + 1e-6
    )

    # Compute adaptive sigma if not provided
    if sigma is None:
        pairwise_dists = torch.norm(
            source_features[:, None] - target_features, dim=2, p=2
        )
        sigma = torch.median(pairwise_dists).detach().item()
        sigma = max(sigma, 1e-3)  # Avoid very small sigma

    # Compute kernel matrices
    if kernel == "rbf":
        K_ss = rbf_kernel(source_features, source_features, sigma)
        K_tt = rbf_kernel(target_features, target_features, sigma)
        K_st = rbf_kernel(source_features, target_features, sigma)
    elif kernel == "linear":
        K_ss = torch.matmul(source_features, source_features.t())
        K_tt = torch.matmul(target_features, target_features.t())
        K_st = torch.matmul(source_features, target_features.t())
    else:
        raise ValueError("Invalid kernel type. Choose 'rbf' or 'linear'.")

    # Compute normalized MMD loss
    mmd_loss = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()

    return mmd_loss


class UNet_MMD(nn.Module):
    def __init__(self):
        super(UNet_MMD, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(2, 64)  # 1 variable + 1 elevation
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, variable, elevation, target_variable=None, target_elevation=None):
        assert variable.shape[2:] == elevation.shape[2:] == (128, 128)

        # Concatenate source input
        x = torch.cat((variable, elevation), dim=1)  # [B, 2, 128, 128]
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)
        b = self.bottleneck(p3)

        # Optional: Compute MMD loss from encoder3 outputs
        mmd_loss = 0
        if target_variable is not None and target_elevation is not None:
            target_x = torch.cat((target_variable, target_elevation), dim=1)
            te1 = self.encoder1(target_x)
            tp1 = self.pool(te1)
            te2 = self.encoder2(tp1)
            tp2 = self.pool(te2)
            te3 = self.encoder3(tp2)
            mmd_loss = compute_mmd(e3.flatten(1), te3.flatten(1))

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output(d1), mmd_loss
