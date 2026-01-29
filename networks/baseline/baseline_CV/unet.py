import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils import data

# 

def conv(in_planes, output_channels, kernel_size, stride, dropout_rate):
    return nn.Sequential(
        nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate)
    )

def deconv(input_channels, output_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                           stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )

def output_layer(input_channels, output_channels, kernel_size, stride, dropout_rate):
    return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                     stride=stride, padding=(kernel_size - 1) // 2)

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dropout_rate=0.5):
        super(U_net, self).__init__()
        self.input_channels = input_channels

        # Encoder
        # H/2
        self.conv1 = conv(input_channels, 64, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        # H/4
        self.conv2 = conv(64, 128, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        # H/8
        self.conv3 = conv(128, 256, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        # H/8 (No downsampling here in original code logic for conv3_1)
        self.conv3_1 = conv(256, 256, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

        # Decoder
        # Upsample to H/4
        self.deconv2 = deconv(256, 64)
        # Upsample to H/2
        # Input channels = 128 (from conv2) + 64 (from deconv2) = 192
        self.deconv1 = deconv(192, 32)
        # Upsample to H
        # Input channels = 64 (from conv1) + 32 (from deconv1) = 96
        self.deconv0 = deconv(96, 16)
        
        # Output Layer
        # Input channels = input_channels (skip connection from input) + 16 (from deconv0)
        self.output_layer = output_layer(16 + input_channels, output_channels,
                                        kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            out: Output tensor of shape (B, out_C, H, W)
        """
        # x shape: (B, C, H, W)
        
        # Encoder path (downsampling)
        out_conv1 = self.conv1(x)                  # (B, 64, H/2, W/2)
        out_conv2 = self.conv2(out_conv1)          # (B, 128, H/4, W/4)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))  # (B, 256, H/8, W/8)

        # Decoder path (upsampling)
        out_deconv2 = self.deconv2(out_conv3)      # (B, 64, H/4, W/4)
        
        # Skip connection 2
        concat2 = torch.cat((out_conv2, out_deconv2), 1)  # (B, 128+64=192, H/4, W/4)
        out_deconv1 = self.deconv1(concat2)        # (B, 32, H/2, W/2)
        
        # Skip connection 1
        concat1 = torch.cat((out_conv1, out_deconv1), 1)  # (B, 64+32=96, H/2, W/2)
        out_deconv0 = self.deconv0(concat1)        # (B, 16, H, W)
        
        # Skip connection 0 (Input)
        concat0 = torch.cat((x, out_deconv0), 1)   # (B, in_C+16, H, W)
        out = self.output_layer(concat0)           # (B, out_C, H, W)

        return out

if __name__ == "__main__":
    # Minimal sanity check.
    B = 1
    in_channels = 99
    out_channels = 93
    height = 360
    width = 720
    
    # Initialize model
    model = U_net(input_channels=in_channels, output_channels=out_channels)
    
    # Synthetic input (B, C, H, W)
    input_tensor = torch.randn(B, in_channels, height, width)
    print(f"Input shape: {input_tensor.shape}")

    # Forward-only check (no_grad)
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Output shape: {output.shape}")
    
    expected_shape = (B, out_channels, height, width)
    assert output.shape == expected_shape, f"Mismatch: {output.shape} vs {expected_shape}"
    print("Sanity check passed.")