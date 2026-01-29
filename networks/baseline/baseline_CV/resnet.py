import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

# 

class Resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, dropout_rate, res=True):
        super(Resblock, self).__init__()
        # Layer 1: Changes dimension if input_channels != hidden_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        # Layer 2: Maintains dimension
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        self.res = res

    def forward(self, x):
        out = self.layer1(x)
        if self.res:
            # Note: This requires x and out to have same shape (channels and resolution)
            out = self.layer2(out) + x
        else:
            out = self.layer2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels, dropout_rate=0.5):
        super(ResNet, self).__init__()
        
        # Initial projection
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        # Stacking Residual Blocks
        # Note: This architecture maintains spatial resolution (H, W) throughout
        # while increasing channel depth. This is memory intensive.
        layers = [Resblock(64, 64, dropout_rate) for _ in range(3)]
        
        # 64 -> 128
        layers += [Resblock(64, 128, dropout_rate, res=False)] + [Resblock(128, 128, dropout_rate) for _ in range(3)]
        
        # 128 -> 256
        layers += [Resblock(128, 256, dropout_rate, res=False)] + [Resblock(256, 256, dropout_rate) for _ in range(5)]
        
        # 256 -> 512
        layers += [Resblock(256, 512, dropout_rate, res=False)] + [Resblock(512, 512, dropout_rate) for _ in range(2)]
        
        self.middle_layer = nn.Sequential(*layers)
        
        # Final projection to output channels
        self.output_layer = nn.Conv2d(512, output_channels, kernel_size=3, padding=1)

        # Initialize batch normalization layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            out: Output tensor of shape (B, out_C, H, W)
        """
        # (B, C, H, W) -> (B, 64, H, W)
        out = self.input_layer(x)
        
        # (B, 64, H, W) -> ... -> (B, 512, H, W)
        out = self.middle_layer(out)
        
        # (B, 512, H, W) -> (B, out_C, H, W)
        out = self.output_layer(out)
        
        return out

if __name__ == "__main__":
    # Minimal sanity check.
    B = 1
    in_channels = 99
    out_channels = 93
    height = 360
    width = 720
    
    # Initialize model
    model = ResNet(input_channels=in_channels, output_channels=out_channels, dropout_rate=0.5)
    
    # Synthetic input (B, C, H, W)
    input_tensor = torch.randn(B, in_channels, height, width)
    print(f"Input shape: {input_tensor.shape}")

    # Forward-only check (no_grad). This model can be memory-heavy at 360x720.
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Output shape: {output.shape}")
    
    expected_shape = (B, out_channels, height, width)
    assert output.shape == expected_shape, f"Mismatch: {output.shape} vs {expected_shape}"
    print("Sanity check passed.")