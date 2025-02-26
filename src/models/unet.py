import torch
import torch.nn as nn
import pytorch_lightning as pl

class UNet8xBaseline(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super(UNet8xBaseline, self).__init__()
        self.lr = lr
        self.criterion = nn.MSELoss()

        # Elevation Downsampling
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.encoder1 = self.conv_block(65, 64)  
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64)

        self.upconv_final = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        )
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, temperature, elevation):
        elevation_downsampled = self.downsample_elevation(elevation)
        x = torch.cat((temperature, elevation_downsampled), dim=1)  

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.decoder3(torch.cat((self.upconv3(b), e3), dim=1))
        d2 = self.decoder2(torch.cat((self.upconv2(d3), e2), dim=1))
        d1 = self.decoder1(torch.cat((self.upconv1(d2), e1), dim=1))
        
        return self.output(self.upconv_final(d1))

    def training_step(self, batch, batch_idx):
        temperature, elevation, target = batch
        output = self.forward(temperature, elevation)
        loss = self.criterion(output, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        temperature, elevation, target = batch
        output = self.forward(temperature, elevation)
        loss = self.criterion(output, target)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]