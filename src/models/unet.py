import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
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
            nn.ReLU(inplace=True)
        )
    
    def forward(self, variable, elevation):  
        # Check input resolution
        assert variable.shape[2:] == elevation.shape[2:] == (128, 128), \
            f"Inputs must be same shape (128x128), got {variable.shape[2:]} and {elevation.shape[2:]}"
        
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
            nn.ReLU(inplace=True)
        )
    
    def forward(self, variable, elevation):  
        # Check input resolution
        assert variable.shape[2:] == elevation.shape[2:] == (128, 128), \
            f"Inputs must be same shape (128x128), got {variable.shape[2:]} and {elevation.shape[2:]}"
        
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
        
    def conv_block(self, in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)
        )
    
    def forward(self, variable, elevation):  
        # Check input resolution
        assert variable.shape[2:] == elevation.shape[2:] == (128, 128), \
            f"Inputs must be same shape (128x128), got {variable.shape[2:]} and {elevation.shape[2:]}"
        
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
        
    def conv_block(self, in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)
        )
    
    def forward(self, variable, elevation):  
        # Check input resolution
        assert variable.shape[2:] == elevation.shape[2:] == (128, 128), \
            f"Inputs must be same shape (128x128), got {variable.shape[2:]} and {elevation.shape[2:]}"
        
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
            nn.ReLU(inplace=True)
        )
    
    def forward(self, variable, elevation): 
        # Apply multiplicative Gaussian noise in training mode
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(variable) * self.noise_std  # Sample noise
            variable_noisy = variable * (1 + noise)  # Multiplicative noise
        else:
            variable_noisy = variable
 
        # Check input resolution
        assert variable_noisy.shape[2:] == elevation.shape[2:] == (128, 128), \
            f"Inputs must be same shape (128x128), got {variable_noisy.shape[2:]} and {elevation.shape[2:]}"
        
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
        
    def conv_block(self, in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)
        )
    
    def forward(self, variable, elevation): 
        # Apply multiplicative Gaussian noise in training mode
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(variable) * self.noise_std  # Sample noise
            variable_noisy = variable * (1 + noise)  # Multiplicative noise
        else:
            variable_noisy = variable
 
        # Check input resolution
        assert variable_noisy.shape[2:] == elevation.shape[2:] == (128, 128), \
            f"Inputs must be same shape (128x128), got {variable_noisy.shape[2:]} and {elevation.shape[2:]}"
        
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
        self.input_log_var = nn.Parameter(torch.zeros(1))  # log variance for input noise
        
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
            nn.ReLU(inplace=True)
        )
    
    def forward(self, variable, elevation): 
        # Apply multiplicative Gaussian noise in training mode
        if self.training:
            input_std = torch.exp(0.5 * self.input_log_var)  # Compute std from log variance
            noise = torch.randn_like(variable) * input_std  # Sample noise
            variable_noisy = variable * (1 + noise)  # Multiplicative noise
        else:
            variable_noisy = variable
 
        # Check input resolution
        assert variable_noisy.shape[2:] == elevation.shape[2:] == (128, 128), \
            f"Inputs must be same shape (128x128), got {variable_noisy.shape[2:]} and {elevation.shape[2:]}"
        
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
