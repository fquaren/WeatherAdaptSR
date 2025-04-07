import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np


class UNet8x(nn.Module):
    def __init__(self):
        super(UNet8x, self).__init__()
        
        # Elevation Downsampling Block (to match variable resolution)
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 2x downsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 2x downsampling (total 4x)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 2x downsampling (total 8x)
            nn.ReLU(inplace=True),
        )

        # Define encoding layers
        self.encoder1 = self.conv_block(65, 64)  # 32 (from elevation) + 1 (variable)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64)
        
        # Additional upsampling layers for 8x resolution
        self.upconv_final1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 2x
        self.upconv_final2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 4x
        self.upconv_final3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 8x
        self.output = nn.Conv2d(64, 1, kernel_size=1)  # Final output layer
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, variable, elevation):  
        # Downsample elevation data to match variable resolution
        elevation_downsampled = self.downsample_elevation(elevation)
        
        # # Upscale elevation data to match variable resolution
        # elevation_downsampled = self.upscale_elevation(elevation)

        # Check dimensions
        assert variable.shape[2:] == elevation_downsampled.shape[2:], \
            f"Selected variable and elevation dimensions do not match, {variable.shape[2:], elevation_downsampled.shape[2:]}"

        # Concatenate the two inputs
        x = torch.cat((variable, elevation_downsampled), dim=1)  
        
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
        d3 = torch.cat((d3, e3), dim=1)  # Skip connection
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)  # Skip connection
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)  # Skip connection
        d1 = self.decoder1(d1)
        
        # Additional upsampling for 8x resolution
        d_final1 = self.upconv_final1(d1)
        d_final2 = self.upconv_final2(d_final1)
        d_final3 = self.upconv_final3(d_final2)
        return self.output(d_final3)


class UNet8x_BN(nn.Module):
    def __init__(self):
        super(UNet8x_BN, self).__init__()
        
        # Elevation Downsampling Block (to match variable resolution)
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Define encoding layers
        self.encoder1 = self.conv_block(65, 64)  
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64)
        
        # Additional upsampling layers for 8x resolution
        self.upconv_final1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, variable, elevation):  
        # Downsample elevation data
        elevation_downsampled = self.downsample_elevation(elevation)

        # Check dimensions
        assert variable.shape[2:] == elevation_downsampled.shape[2:], \
            f"Dimension mismatch: {variable.shape[2:]} vs {elevation_downsampled.shape[2:]}"

        # Concatenate inputs
        x = torch.cat((variable, elevation_downsampled), dim=1)  
        
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
        
        # Additional upsampling for 8x resolution
        d_final1 = self.upconv_final1(d1)
        d_final2 = self.upconv_final2(d_final1)
        d_final3 = self.upconv_final3(d_final2)
        return self.output(d_final3)


class UNet8x_DO(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(UNet8x_DO, self).__init__()
        
        # Elevation Downsampling Block (to match variable resolution)
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Define encoding layers
        self.encoder1 = self.conv_block(65, 64, dropout_prob=0)  
        self.encoder2 = self.conv_block(64, 128, dropout_prob=0)
        self.encoder3 = self.conv_block(128, 256, dropout_prob)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, dropout_prob)
        
        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 256, 256, dropout_prob)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 128, 128, dropout_prob=0)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64, dropout_prob=0)
        
        # Additional upsampling layers for 8x resolution
        self.upconv_final1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, variable, elevation):  
        # Downsample elevation data
        elevation_downsampled = self.downsample_elevation(elevation)

        # Check dimensions
        assert variable.shape[2:] == elevation_downsampled.shape[2:], \
            f"Dimension mismatch: {variable.shape[2:]} vs {elevation_downsampled.shape[2:]}"

        # Concatenate inputs
        x = torch.cat((variable, elevation_downsampled), dim=1)  
        
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
        
        # Additional upsampling for 8x resolution
        d_final1 = self.upconv_final1(d1)
        d_final2 = self.upconv_final2(d_final1)
        d_final3 = self.upconv_final3(d_final2)
        return self.output(d_final3)

class UNet8x_DO_BN(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(UNet8x_DO_BN, self).__init__()
        
        # Elevation Downsampling Block (to match variable resolution)
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Define encoding layers
        self.encoder1 = self.conv_block(65, 64, dropout_prob=0)  
        self.encoder2 = self.conv_block(64, 128, dropout_prob=0)
        self.encoder3 = self.conv_block(128, 256, dropout_prob)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, dropout_prob)
        
        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 256, 256, dropout_prob)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 128, 128, dropout_prob=0)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64, dropout_prob=0)
        
        # Additional upsampling layers for 8x resolution
        self.upconv_final1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, variable, elevation):  
        # Downsample elevation data
        elevation_downsampled = self.downsample_elevation(elevation)

        # Check dimensions
        assert variable.shape[2:] == elevation_downsampled.shape[2:], \
            f"Dimension mismatch: {variable.shape[2:]} vs {elevation_downsampled.shape[2:]}"

        # Concatenate inputs
        x = torch.cat((variable, elevation_downsampled), dim=1)  
        
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
        
        # Additional upsampling for 8x resolution
        d_final1 = self.upconv_final1(d1)
        d_final2 = self.upconv_final2(d_final1)
        d_final3 = self.upconv_final3(d_final2)
        return self.output(d_final3)


class UNet8x_Noise(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(UNet8x_Noise, self).__init__()

        # Input noise parameters (learned)
        self.input_log_var = nn.Parameter(torch.zeros(1))  # log variance for input noise

        # Elevation Downsampling Block
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Define encoding layers
        self.encoder1 = self.conv_block(65, 64, dropout_prob=0)  
        self.encoder2 = self.conv_block(64, 128, dropout_prob=0)
        self.encoder3 = self.conv_block(128, 256, dropout_prob)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, dropout_prob)
        
        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 256, 256, dropout_prob)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 128, 128, dropout_prob=0)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64, dropout_prob=0)
        
        # Additional upsampling layers for 8x resolution
        self.upconv_final1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, variable, elevation):  
        # If training, add learnable Gaussian noise to input
        # Check if the model is in training mode
        if self.training:
            input_std = torch.exp(0.5 * self.input_log_var)  # Compute std from log variance
            noise = torch.randn_like(variable) * input_std  # Sample noise
            variable_noisy = variable + noise  # Apply noise
        else:
            variable_noisy = variable
        
        # Downsample elevation data
        elevation_downsampled = self.downsample_elevation(elevation)

        # Check dimensions
        assert variable_noisy.shape[2:] == elevation_downsampled.shape[2:], \
            f"Dimension mismatch: {variable_noisy.shape[2:]} vs {elevation_downsampled.shape[2:]}"

        # Concatenate inputs
        x = torch.cat((variable_noisy, elevation_downsampled), dim=1)  
        
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
        
        # Additional upsampling for 8x resolution
        d_final1 = self.upconv_final1(d1)
        d_final2 = self.upconv_final2(d_final1)
        d_final3 = self.upconv_final3(d_final2)
        return self.output(d_final3)
    

class UNet8x_Noise(nn.Module):
    def __init__(self, dropout_prob=0.3, noise_std=0.1):
        super(UNet8x_Noise, self).__init__()

        self.noise_std = noise_std  # Non-trainable parameter

        # Learnable variance for Gaussian noise
        self.input_log_var = nn.Parameter(torch.zeros(1))  

        # Elevation Downsampling Block
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Define encoding layers
        self.encoder1 = self.conv_block(65, 64, dropout_prob=0)  
        self.encoder2 = self.conv_block(64, 128, dropout_prob=0)
        self.encoder3 = self.conv_block(128, 256, dropout_prob)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, dropout_prob)
        
        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 256, 256, dropout_prob)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 128, 128, dropout_prob=0)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64, dropout_prob=0)
        
        # Additional upsampling layers for 8x resolution
        self.upconv_final1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )

    def add_gaussian_multiplicative_noise(self, x):
        """
        Apply Gaussian multiplicative noise to input tensor.
        
        Parameters:
        - x: torch.Tensor, input data
        
        Returns:
        - torch.Tensor, noised data
        """
        if self.training:  # Apply noise only during training
            noise = torch.randn_like(x) * self.noise_std + 1  # Multiplicative noise: N(1, noise_std)
            return x * noise
        else:
            return x  # No noise during evaluation
    
    def forward(self, variable, elevation):  
        if self.training:
            variable_noisy = self.add_gaussian_multiplicative_noise(variable)
        else:
            variable_noisy = variable
        
        # Downsample elevation data
        elevation_downsampled = self.downsample_elevation(elevation)

        # Ensure matching dimensions
        assert variable_noisy.shape[2:] == elevation_downsampled.shape[2:], \
            f"Dimension mismatch: {variable_noisy.shape[2:]} vs {elevation_downsampled.shape[2:]}"

        # Concatenate inputs
        x = torch.cat((variable_noisy, elevation_downsampled), dim=1)  
        
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
        
        # Additional upsampling for 8x resolution
        d_final1 = self.upconv_final1(d1)
        d_final2 = self.upconv_final2(d_final1)
        d_final3 = self.upconv_final3(d_final2)
        return self.output(d_final3)


# Multi source domain adaptation UNet
class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input

class UNet8x_MDAN(nn.Module):
    def __init__(self, num_domains=11):  # Number of source domains
        super(UNet8x_MDAN, self).__init__()

        # Elevation Downsampling Block (to match variable resolution)
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 2x downsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 2x downsampling (total 4x)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 2x downsampling (total 8x)
            nn.ReLU(inplace=True),
        )

        # Define encoding layers
        self.encoder1 = self.conv_block(65, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64)

        # Additional upsampling for 8x resolution
        self.upconv_final1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.output = nn.Conv2d(64, 1, kernel_size=1)

        # Domain Classifiers
        self.num_domains = num_domains
        self.domain_classifiers = nn.ModuleList([nn.Linear(2048, 2) for _ in range(num_domains)])

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, variable, elevation, domain_idx=None):
        # Downsample elevation data
        elevation_downsampled = self.downsample_elevation(elevation)
        
        # Check dimensions
        assert variable.shape[2:] == elevation_downsampled.shape[2:], \
            f"Selected variable and elevation dimensions do not match, {variable.shape[2:], elevation_downsampled.shape[2:]}"

        # Concatenate inputs
        x = torch.cat((variable, elevation_downsampled), dim=1)

        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Flatten bottleneck features for domain classification
        b_flat = b.view(b.shape[0], -1)

        # Domain Classification (if domain label is provided)
        domain_preds = None
        if domain_idx is not None:
            grl = GradientReversalLayer.apply(b_flat)  # Apply Gradient Reversal Layer
            domain_preds = self.domain_classifiers[domain_idx](grl)

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

        # Additional upsampling
        d_final1 = self.upconv_final1(d1)
        d_final2 = self.upconv_final2(d_final1)
        d_final3 = self.upconv_final3(d_final2)

        return self.output(d_final3), domain_preds


# Probabilistic learning UNet
class UNet8x_Engression(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(UNet8x_Engression, self).__init__()

        # Elevation Downsampling Block (to match variable resolution)
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Define encoding layers
        self.encoder1 = self.conv_block(65, 64, dropout_prob)
        self.encoder2 = self.conv_block(64, 128, dropout_prob)
        self.encoder3 = self.conv_block(128, 256, dropout_prob)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, dropout_prob)

        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 256, 256, dropout_prob)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 128, 128, dropout_prob)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64, dropout_prob)

        # Additional upsampling layers for 8x resolution
        self.upconv_final1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv_final3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # Output layer: 2 channels (mean and log variance)
        self.output = nn.Conv2d(64, 2, kernel_size=1)  # Two outputs: mean and log variance

    def conv_block(self, in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )

    def forward(self, variable, elevation):
        # Downsample elevation data
        elevation_downsampled = self.downsample_elevation(elevation)

        # Check dimensions
        assert variable.shape[2:] == elevation_downsampled.shape[2:], \
            f"Dimension mismatch: {variable.shape[2:]} vs {elevation_downsampled.shape[2:]}"

        # Concatenate inputs
        x = torch.cat((variable, elevation_downsampled), dim=1)

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

        # Additional upsampling for 8x resolution
        d_final1 = self.upconv_final1(d1)
        d_final2 = self.upconv_final2(d_final1)
        d_final3 = self.upconv_final3(d_final2)

        # Output layer: Mean and log variance
        output = self.output(d_final3)  
        mean, log_var = output[:, 0:1], output[:, 1:2]  # Split into mean and log variance

        # Convert log variance to standard deviation (ensures positivity)
        std = torch.exp(0.5 * log_var)

        # Sampled prediction: y = mean + std * noise (only during training)
        epsilon = torch.randn_like(std) if self.training else 0
        sampled_output = mean + std * epsilon

        return sampled_output, mean, std  # Return all for loss computation