import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    A block of two sequential 3x3 convolutions, each followed by
    Batch Normalization and a ReLU activation.
    (CONV -> BN -> ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    A standard UNet architecture.

    The model expects and produces tensors of the same spatial resolution.
    The number of input channels is determined by your input variables
    (e.g., 12 dynamic + 2 static = 14).
    The number of output channels is 1 (precipitation).
    """

    def __init__(self, in_channels, out_channels, n_features_base=64):
        super(UNet, self).__init__()

        # Encoder (Downsampling Path)
        self.inc = DoubleConv(in_channels, n_features_base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(n_features_base, n_features_base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(n_features_base * 2, n_features_base * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(n_features_base * 4, n_features_base * 8))

        # Bottleneck
        factor = 2
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(n_features_base * 8, (n_features_base * 16) // factor))

        # Decoder (Upsampling Path)
        self.up1 = nn.ConvTranspose2d(
            n_features_base * 16 // factor, n_features_base * 8 // factor, kernel_size=2, stride=2
        )
        # --- MODIFICATION 1 ---
        # Input channels = skip (n_features_base * 8) + upsample (n_features_base * 8 // factor)
        self.conv1 = DoubleConv(n_features_base * 8 + n_features_base * 8 // factor, n_features_base * 8 // factor)

        self.up2 = nn.ConvTranspose2d(
            n_features_base * 8 // factor, n_features_base * 4 // factor, kernel_size=2, stride=2
        )
        # --- MODIFICATION 2 ---
        # Input channels = skip (n_features_base * 4) + upsample (n_features_base * 4 // factor)
        self.conv2 = DoubleConv(n_features_base * 4 + n_features_base * 4 // factor, n_features_base * 4 // factor)

        self.up3 = nn.ConvTranspose2d(
            n_features_base * 4 // factor, n_features_base * 2 // factor, kernel_size=2, stride=2
        )
        # --- MODIFICATION 3 ---
        # Input channels = skip (n_features_base * 2) + upsample (n_features_base * 2 // factor)
        self.conv3 = DoubleConv(n_features_base * 2 + n_features_base * 2 // factor, n_features_base * 2 // factor)

        self.up4 = nn.ConvTranspose2d(n_features_base * 2 // factor, n_features_base, kernel_size=2, stride=2)
        # --- MODIFICATION 4 ---
        # Input channels = skip (n_features_base) + upsample (n_features_base)
        self.conv4 = DoubleConv(n_features_base + n_features_base, n_features_base)
        # --- END MODIFICATIONS ---

        # Final 1x1 convolution to map to the desired output channels
        self.outc = nn.Conv2d(n_features_base, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C_in, 200, 200)

        # Encoder
        x1 = self.inc(x)  # (B, 64, 200, 200)
        x2 = self.down1(x1)  # (B, 128, 100, 100)
        x3 = self.down2(x2)  # (B, 256, 50, 50)
        x4 = self.down3(x3)  # (B, 512, 25, 25)

        # Adjusting for non-power-of-2 input (200 -> 100 -> 50 -> 25)
        # The next max pool will be (25 -> 12)
        x5 = self.down4(x4)  # (B, 512, 12, 12)

        # Decoder
        x = self.up1(x5)  # (B, 256, 24, 24)
        # --- Skip Connection 1 ---
        # x4 is (25, 25), x is (24, 24). We must crop x4.
        x4 = self.crop(x4, x)
        x = torch.cat([x4, x], dim=1)  # (B, 768, 24, 24)
        x = self.conv1(x)  # (B, 256, 24, 24)

        x = self.up2(x)  # (B, 128, 48, 48)
        # --- Skip Connection 2 ---
        # x3 is (50, 50), x is (48, 48). Crop x3.
        x3 = self.crop(x3, x)
        x = torch.cat([x3, x], dim=1)  # (B, 384, 48, 48)
        x = self.conv2(x)  # (B, 128, 48, 48)

        x = self.up3(x)  # (B, 64, 96, 96)
        # --- Skip Connection 3 ---
        # x2 is (100, 100), x is (96, 96). Crop x2.
        x2 = self.crop(x2, x)
        x = torch.cat([x2, x], dim=1)  # (B, 192, 96, 96)
        x = self.conv3(x)  # (B, 64, 96, 96)

        x = self.up4(x)  # (B, 64, 192, 192)
        # --- Skip Connection 4 ---
        # x1 is (200, 200), x is (192, 192). Crop x1.
        x1 = self.crop(x1, x)
        x = torch.cat([x1, x], dim=1)  # (B, 128, 192, 192)
        x = self.conv4(x)  # (B, 64, 192, 192)

        # Final output convolution
        logits = self.outc(x)  # (B, C_out, 192, 192)

        # --- Final Padding ---
        # The output is (192, 192) but we need (200, 200).
        # We must pad the output to match the input size.
        # (Pad left, Pad right, Pad top, Pad bottom)
        # (200 - 192) = 8. We need 4 on each side.
        final_logits = F.pad(logits, (4, 4, 4, 4), "constant", 0)

        return final_logits

    def crop(self, skip_tensor, up_tensor):
        """
        Crops the skip connection tensor to match the spatial dimensions
        of the up-sampled tensor.
        """
        _, _, H, W = up_tensor.shape
        _, _, H_skip, W_skip = skip_tensor.shape

        # Calculate the top-left corner for cropping
        h_diff = (H_skip - H) // 2
        w_diff = (W_skip - W) // 2

        # Perform the crop
        return skip_tensor[:, :, h_diff : h_diff + H, w_diff : w_diff + W]


if __name__ == "__main__":
    """
    A simple test to verify the model's tensor shapes.
    """
    print("Running UNet shape test...")

    # Your dataset uses 9 dynamic + 2 static = 11 channels
    IN_CHANNELS = 11

    # Batch size = 4
    # H, W = 200, 200
    test_tensor = torch.randn(4, IN_CHANNELS, 200, 200)

    model = UNet(in_channels=IN_CHANNELS, out_channels=1)

    output = model(test_tensor)

    print(f"Input tensor shape:  {test_tensor.shape}")
    print(f"Output tensor shape: {output.shape}")

    # Check if the output shape is correct
    assert output.shape == (4, 1, 200, 200)

    print("Test passed: Output shape is correct.")

    # Also test the crop function with odd/even mismatch
    print("\nRunning crop function test...")
    skip = torch.randn(1, 10, 25, 25)  # e.g., x4 from encoder
    up = torch.randn(1, 20, 12, 12)  # e.g., x5 max-pooled
    up_t = torch.randn(1, 5, 24, 24)  # e.g., x (transposed)

    cropped_skip = model.crop(skip, up_t)
    print(f"Cropping (25, 25) to (24, 24)... Result shape: {cropped_skip.shape}")
    assert cropped_skip.shape == (1, 10, 24, 24)
    print("Test passed: Crop function is correct.")
