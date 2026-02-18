import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class EDSRModel(nn.Module):
    """
    Enhanced Deep Residual Networks Super Resolution network for non-integer scaling (e.g., 2.5x).
    Extracts features in low-resolution space, then performs bilinear interpolation
    on deep feature maps, followed by high-resolution convolution.
    """

    def __init__(self, dynamic_in_channels=9, static_in_channels=2, out_channels=1, features=128, num_res_blocks=6):
        super().__init__()

        # 1. Low-resolution feature extraction (Dynamic variables only)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(dynamic_in_channels, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(features) for _ in range(num_res_blocks)])

        self.lr_conv = nn.Conv2d(features, features, kernel_size=3, padding=1)

        # 2. High-resolution fusion (Static variables added here)
        # We concatenate the upsampled deep features with the static variables
        self.hr_fusion = nn.Sequential(
            nn.Conv2d(features + static_in_channels, features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_dynamic, x_static):
        # Extract features in low-res space
        feat = self.initial_conv(x_dynamic)
        feat_res = self.res_blocks(feat)
        feat = self.lr_conv(feat_res) + feat

        # Upsample deep features to match static high-res tensor dimensions
        _, _, H_hr, W_hr = x_static.shape
        feat_upsampled = F.interpolate(feat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

        # Fuse with static variables directly in high-res space
        fused = torch.cat([feat_upsampled, x_static], dim=1)

        return self.hr_fusion(fused)


if __name__ == "__main__":
    print("Running ClimateSRModel shape and forward pass test...")

    # Define dimensions based on the 2.5x super resolution task
    batch_size = 4
    dynamic_channels = 9
    static_channels = 2

    # Low resolution input (200 / 2.5 = 80)
    lr_size = 80
    # High resolution target
    hr_size = 200

    # Initialize random tensors simulating your batch data
    x_dynamic = torch.randn(batch_size, dynamic_channels, lr_size, lr_size)
    x_static = torch.randn(batch_size, static_channels, hr_size, hr_size)

    # Instantiate the model
    model = EDSRModel(
        dynamic_in_channels=dynamic_channels,
        static_in_channels=static_channels,
        out_channels=1,
        features=64,  # Reduced for a quick forward pass test
        num_res_blocks=2,
    )

    # Perform forward pass
    output = model(x_dynamic, x_static)

    print(f"Dynamic input (LR) shape: {x_dynamic.shape}")
    print(f"Static input (HR) shape:  {x_static.shape}")
    print(f"Output tensor (HR) shape: {output.shape}")

    # Verify the output matches the expected high-resolution grid
    assert output.shape == (
        batch_size,
        1,
        hr_size,
        hr_size,
    ), "Error: Output spatial dimensions do not match target high-resolution grid."

    print(
        "Test passed: Low-resolution features successfully interpolated and fused with high-resolution static fields."
    )
