import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.backends.cudnn as cudnn
import numpy as np
import random
from functools import partial
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# --- Utilities ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(self, img_size=None, patch_size=8, in_c=13, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # (h, w)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        x = self.projection(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AdativeFourierNeuralOperator(nn.Module):
    def __init__(self, dim, h=14, w=14, fno_blocks=4, fno_bias=False, fno_softshrink=0.0):
        super(AdativeFourierNeuralOperator, self).__init__()
        self.hidden_size = dim
        self.h = h
        self.w = w
        self.num_blocks = fno_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0, "Hidden size must be divisible by number of FNO blocks"

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        
        self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1) if fno_bias else None
        self.softshrink = fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum('...bd, bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        if self.bias:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros_like(x)

        x = x.reshape(B, self.h, self.w, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = x.real
        x_imag = x.imag

        x_real_new = F.relu(self.multiply(x_real, self.w1[0]) - self.multiply(x_imag, self.w1[1]) + self.b1[0], inplace=True)
        x_imag_new = F.relu(self.multiply(x_real, self.w1[1]) + self.multiply(x_imag, self.w1[0]) + self.b1[1], inplace=True)
        
        x_real = self.multiply(x_real_new, self.w2[0]) - self.multiply(x_imag_new, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real_new, self.w2[1]) + self.multiply(x_imag_new, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x + bias

class FourierNetBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 h=14, w=14, fno_blocks=4, fno_bias=False, fno_softshrink=0.0):
        super(FourierNetBlock, self).__init__()
        self.normlayer1 = norm_layer(dim)
        self.filter = AdativeFourierNeuralOperator(dim, h=h, w=w, fno_blocks=fno_blocks,
                                                   fno_bias=fno_bias, fno_softshrink=fno_softshrink)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.normlayer2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.filter(self.normlayer1(x)))
        x = x + self.drop_path(self.mlp(self.normlayer2(x)))
        return x

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(num_residual_layers)])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2, out_channels=num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = F.relu(self._conv_1(inputs))
        x = F.relu(self._conv_2(x))
        x = self._conv_3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        return self._conv_trans_2(x)

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0: groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm: y = self.activate(self.norm(y))
        return y

class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers: y += layer(x)
        return y

# --- FPG block (supports patch_size=8 and flexible reconstruction) ---
class FPG(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_channels=20, out_channels=20, input_frames=20,
                 embed_dim=768, depth=12, mlp_ratio=4., norm_layer=None):
        super(FPG, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = input_frames
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Ensure patch_size matches the upsampling logic (Total factor 8: 2*2*2)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.h = self.patch_embed.grid_size[0]
        self.w = self.patch_embed.grid_size[1]

        self.blocks = nn.ModuleList([FourierNetBlock(
            dim=embed_dim, mlp_ratio=mlp_ratio, act_layer=nn.GELU, norm_layer=norm_layer, h=self.h, w=self.w)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Modified Decoder: Upsample x8 (2*2*2) to match patch_size=8
        self.linearprojection = nn.Sequential(OrderedDict([
            ('transposeconv1', nn.ConvTranspose2d(embed_dim, out_channels * 16, kernel_size=(2, 2), stride=(2, 2))),
            ('act1', nn.Tanh()),
            ('transposeconv2', nn.ConvTranspose2d(out_channels * 16, out_channels * 4, kernel_size=(2, 2), stride=(2, 2))),
            ('act2', nn.Tanh()),
            # Last layer projects to out_channels
            ('transposeconv3', nn.ConvTranspose2d(out_channels * 4, out_channels, kernel_size=(2, 2), stride=(2, 2)))
        ]))
        
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks: x = blk(x)
        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(x, [-1, self.embed_dim, self.h, self.w])
        return x

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.forward_features(x)
        x = self.linearprojection(x)
        # Ensure output shape is correct
        x = x.reshape(B, T, -1, H, W) 
        return x

# --- DST block (supports configurable output channels) ---
class DST(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, num_hiddens=128, res_layers=2, res_units=32,
                 embedding_nums=512, embedding_dim=64, commitment_cost=0.25):
        super(DST, self).__init__()
        self.embedding_dim = embedding_dim
        self._encoder = Encoder(in_channel, num_hiddens, res_layers, res_units)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        self._vq_vae = VectorQuantizerEMA(embedding_nums, embedding_dim, commitment_cost, decay=0.99)
        # Decoder outputs out_channel
        self._decoder = Decoder(embedding_dim, num_hiddens, res_layers, res_units, out_channels=out_channel)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

class DynamicPropagation(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(DynamicPropagation, self).__init__()
        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker=incep_ker, groups=groups))

        self.enc = nn.ModuleList(enc_layers)
        self.dec = nn.ModuleList(dec_layers)

    def forward(self, input_state):
        B, T, C, H, W = input_state.shape
        input_state = input_state.reshape(B, T*C, H, W)
        skips = []
        hidden_embed = input_state
        for i in range(self.N_T):
            hidden_embed = self.enc[i](hidden_embed)
            if i < self.N_T - 1: skips.append(hidden_embed)

        hidden_embed = self.dec[0](hidden_embed)
        for i in range(1, self.N_T):
            hidden_embed = self.dec[i](torch.cat([hidden_embed, skips[-i]], dim=1))
        
        return hidden_embed.reshape(B, T, C, H, W)

# --- PastNetModel wrapper: (B, C, H, W) -> (B, OutC, H, W) ---
class PastNetModel(nn.Module):
    def __init__(self, in_channels, out_channels, height, width,
                 hid_T=256, N_T=8, incep_ker=[3, 5, 7, 11], groups=8,
                 res_units=32, res_layers=2, embedding_nums=512, embedding_dim=64):
        super(PastNetModel, self).__init__()
        
        # DST: VQ-VAE Encoder/Decoder path
        self.DST_module = DST(in_channel=in_channels,
                              out_channel=out_channels, # Use out_channels here
                              res_units=res_units,
                              res_layers=res_layers,
                              embedding_dim=embedding_dim,
                              embedding_nums=embedding_nums)

        # FPG: Fourier Transformer path
        # Using patch_size=8 to support 360 height (360 is divisible by 8 but not 16)
        self.FPG_module = FPG(img_size=(height, width),
                              patch_size=8,
                              in_channels=in_channels,
                              out_channels=out_channels, # Use out_channels here
                              embed_dim=128,
                              input_frames=1, # T=1
                              depth=1)

        # Dynamic Propagation: Latent evolution
        self.DynamicPro = DynamicPropagation(1*embedding_dim, hid_T, N_T, incep_ker, groups)

    def forward(self, x):
        # Input x: (B, in_C, H, W)
        
        # 1. Expand Time Dimension -> (B, 1, in_C, H, W)
        x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        
        # --- Path A: FPG (Physics-based global features) ---
        # Returns (B, T, out_C, H, W)
        pde_features = self.FPG_module(x)

        # --- Path B: DST + DynamicPropagation (Latent dynamics) ---
        input_features = x.view(B * T, C, H, W)
        encoder_embed = self.DST_module._encoder(input_features)
        z = self.DST_module._pre_vq_conv(encoder_embed)
        _, Latent_embed, _, _ = self.DST_module._vq_vae(z) # (B*T, embed_dim, h', w')

        _, C_, H_, W_ = Latent_embed.shape
        Latent_embed = Latent_embed.reshape(B, T, C_, H_, W_)

        # Evolution in latent space
        hidden_dim = self.DynamicPro(Latent_embed) # (B, T, C_, H_, W_)
        
        # Decode
        B_, T_, C_out, H_, W_ = hidden_dim.shape
        hid = hidden_dim.view(B_ * T_, C_out, H_, W_)
        predicti_feature = self.DST_module._decoder(hid) # (B*T, out_C, H, W)
        predicti_feature = predicti_feature.view(B, T, -1, H, W) # (B, T, out_C, H, W)

        # --- Combine ---
        output = predicti_feature + pde_features
        
        # 2. Squeeze Time Dimension -> (B, out_C, H, W)
        output = output.squeeze(1)
        
        return output

if __name__ == "__main__":
    set_seed(42)

    # Minimal sanity check.
    B = 1
    in_C = 99
    out_C = 93
    H = 360  # 360/8 = 45 (Integer), 360/4 = 90 (Integer) -> OK
    W = 720  

    print(f"Initializing model with Input ({in_C}, {H}, {W}) -> Output ({out_C}, {H}, {W})...")
    
    model = PastNetModel(
        in_channels=in_C,
        out_channels=out_C,
        height=H,
        width=W,
        hid_T=64, # Reduced for memory in demo
        N_T=2,    # Reduced for memory in demo
        embedding_nums=128
    )

    # Synthetic input.
    input_frames = torch.randn(B, in_C, H, W)
    print("Input shape:", input_frames.shape)

    # Forward-only check (no_grad).
    with torch.no_grad():
        output = model(input_frames)

    print("Output shape:", output.shape)
    
    expected_shape = (B, out_C, H, W)
    assert output.shape == expected_shape, f"Mismatch: {output.shape} vs {expected_shape}"
    print("Verification Successful!")