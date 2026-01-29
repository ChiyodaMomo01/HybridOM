import torch
from torch import nn

# --- Building blocks ---
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, output_padding=stride // 2)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker // 2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            dec_layers.append(
                Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


# --- SimVP wrapper (single-step interface) ---
class SimVP(nn.Module):
    def __init__(self, in_channels, out_channels, hid_S=128, hid_T=256, N_S=4, N_T=4, incep_ker=[3, 5, 7, 11], groups=8):
        """
        Modified SimVP for static-like input (B, C, H, W) -> (B, Out_C, H, W) via temporal expansion.
        """
        super(SimVP, self).__init__()
        
        # Model configuration.
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Spatial encoder: maps inputs to latent feature space.
        self.enc = Encoder(in_channels, hid_S, N_S)
        
        # Mid network: processes spatiotemporal features (here we use T=1).
        self.hid = Mid_Xnet(1 * hid_S, hid_T, N_T, incep_ker, groups)
        
        # Spatial decoder: maps latent features back to output space.
        self.dec = Decoder(hid_S, out_channels, N_S)


    def forward(self, x_raw):
        # x_raw shape: (B, in_C, H, W)
        
        # Add a dummy time dimension (T=1).
        # shape: (B, 1, in_C, H, W)
        x = x_raw.unsqueeze(1)
        
        B, T, C, H, W = x.shape  # T=1
        
        # Flatten time into batch for the encoder.
        x_flat = x.view(B * T, C, H, W)
        
        # Encoder output: (B*T, hid_S, H', W') with skip connections.
        embed, skip = self.enc(x_flat)
        _, C_hid, H_, W_ = embed.shape

        # Reshape for mid network: (B, T, C, H, W).
        z = embed.view(B, T, C_hid, H_, W_)
        
        # Mid_Xnet output: (B, T, C_hid, H', W')
        hid_feature = self.hid(z)
        
        # Flatten time into batch for the decoder.
        hid_feature_flat = hid_feature.reshape(B * T, C_hid, H_, W_)
        
        # Decoder output: (B*T, out_C, H, W)
        Y_flat = self.dec(hid_feature_flat, skip)
        
        # Restore shape: (B, T, out_C, H, W).
        Y = Y_flat.reshape(B, T, self.out_channels, H, W)
        
        # Drop the dummy time dimension: (B, out_C, H, W).
        Y = Y.squeeze(1)
        
        return Y


if __name__ == '__main__':
    # Minimal sanity check.
    batch_size = 1
    in_channels = 99
    out_channels = 93
    height = 360
    width = 720
    
    # Keep the config small to reduce memory usage.
    model = SimVP(in_channels=in_channels, out_channels=out_channels, 
                  hid_S=32, hid_T=64, N_S=2, N_T=4)
    
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Use no_grad for a quick forward-only check.
    with torch.no_grad():
        pred = model(x)
        
    print(f"Output shape: {pred.shape}")
    
    expected_shape = (batch_size, out_channels, height, width)
    assert pred.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {pred.shape}"
    print("Sanity check passed.")