import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
# Requires `timm` (e.g., `pip install timm`).
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.to(next(self.parameters()).device)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=(32, 32),
        patch_size=(2, 2),
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=None,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(
            img_size=input_size, patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        grid_size_h, grid_size_w = self.x_embedder.grid_size
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (grid_size_h, grid_size_w))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p_h, p_w = self.x_embedder.patch_size
        h_patches, w_patches = self.x_embedder.grid_size
        
        x = x.reshape(shape=(x.shape[0], h_patches, w_patches, p_h, p_w, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h_patches * p_h, w_patches * p_w))
        return imgs

    def forward(self, x, t):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        c = t 
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, grid_size[0] * grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

#################################################################################
#                               Other Components                                #
#################################################################################

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class ConvSC(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, transpose=False):
        super(ConvSC, self).__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                           padding=1, output_padding=stride-1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        layers = [ConvSC(C_in, C_hid, stride=strides[0])]
        for s in strides[1:]:
            layers.append(ConvSC(C_hid, C_hid, stride=s))
        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
        return x, skips

class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        layers = []
        for s in strides[:-1]:
            layers.append(ConvSC(C_hid, C_hid, stride=s, transpose=True))
        layers.append(ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True))
        self.dec = nn.Sequential(*layers)
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, skip):
        for i in range(len(self.dec)-1):
            hid = self.dec[i](hid)
        hid = self.dec[-1](torch.cat([hid, skip], dim=1))
        return self.readout(hid)

# Wrapper Class
class DitWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, 
                 hid_S=128, N_S=4, N_T=8):
        super(DitWrapper, self).__init__()
        
        # Store configuration.
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Compute latent spatial size (downsampling depends on strides).
        strides = stride_generator(N_S)
        num_stride2_layers = strides[:N_S].count(2)
        self.downsample_factor = 2 ** num_stride2_layers
        self.H_latent = height // self.downsample_factor
        self.W_latent = width // self.downsample_factor
        
        print(f"Initializing DiT Wrapper: Input {height}x{width} -> Latent {self.H_latent}x{self.W_latent}")

        # 1. Encoder
        self.enc = Encoder(in_channels, hid_S, N_S)
        
        # 2. Core DiT
        # Input channels are hid_S (time is flattened and T=1).
        self.dit_block = DiT(
            input_size=(self.H_latent, self.W_latent),
            patch_size=(1, 1),  # Use 1x1 patches to support non-power-of-two latent sizes.
            in_channels=hid_S,
            hidden_size=256,
            depth=N_T,
            num_heads=4,
            mlp_ratio=4.0,
            class_dropout_prob=0.0,
            num_classes=None,
            learn_sigma=False,
        )

        # 3. Decoder
        self.dec = Decoder(hid_S, out_channels, N_S)

    def forward(self, x):
        """
        Args:
            x: (B, in_C, H, W)
        Returns:
            out: (B, out_C, H, W)
        """
        # Add dummy time dimension and flatten: (B*1, C, H, W)
        x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # 2. Encode
        # embed: (B*T, hid_S, H_latent, W_latent)
        embed, skips = self.enc(x_flat)
        skip = skips[0]  # skip connection (U-Net style)

        # 3. DiT Block
        # Use a zero timestep condition (B*T,).
        t = torch.zeros(B * T, device=x.device, dtype=torch.float32)
        
        # DiT forward: (B*T, hid_S, H_latent, W_latent)
        hid = self.dit_block(embed, t)

        # 4. Decode
        # hid: (B*T, hid_S, H_latent, W_latent) -> (B*T, out_C, H, W)
        Y = self.dec(hid, skip)
        
        # Restore shape and drop dummy time dimension.
        Y = Y.reshape(B, T, self.out_channels, H, W)
        Y = Y.squeeze(1)
        
        return Y

if __name__ == '__main__':
    # Minimal sanity check.
    B = 1
    in_channels = 99
    out_channels = 93
    height = 360  # 360 / 8 = 45
    width = 720   # 720 / 8 = 90
    
    print(f"Testing Model with Input ({in_channels}, {height}, {width})...")
    
    # Keep the model small for a quick check.
    model = DitWrapper(
        in_channels=in_channels, 
        out_channels=out_channels,
        height=height,
        width=width,
        hid_S=32,
        N_T=2
    )

    # Synthetic input
    inputs = torch.randn(B, in_channels, height, width)
    print('Inputs shape:', inputs.shape)
    
    # Forward-only check (no_grad)
    with torch.no_grad():
        output = model(inputs)
        
    print('Output shape:', output.shape)
    
    expected_shape = (B, out_channels, height, width)
    assert output.shape == expected_shape, f"Mismatch: {output.shape} vs {expected_shape}"
    print("Sanity check passed.")