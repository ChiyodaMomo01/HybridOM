import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_

# --- Simple DropPath (avoids inplace issues) ---
class SimpleDropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

# --- Helper functions: domain decomposition ---

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def grid_partition(x, grid_size):
    B, H, W, C = x.shape
    x = x.view(B, H // grid_size, grid_size, W // grid_size, grid_size, C)
    grids = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, (H // grid_size) * (W // grid_size), C)
    return grids

def grid_reverse(grids, grid_size, H, W):
    B = int(grids.shape[0] / (grid_size * grid_size))
    x = grids.view(B, grid_size, grid_size, H // grid_size, W // grid_size, -1)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)
    return x

# --- Core components ---

class FeatureMixer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(FeatureMixer, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvFeatureMixer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(ConvFeatureMixer, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop, inplace=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# =========================================================
# Dual-scale attention with optional global/local branches.
# =========================================================
class DualScaleOceanAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 use_global=True, use_local=True):
        super(DualScaleOceanAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        # Branch toggles.
        self.use_global = use_global
        self.use_local = use_local

        # Guard against invalid configuration.
        assert use_global or use_local, "At least one branch (Global or Local) must be True."

        # If both branches are enabled, head count must be even for a clean split.
        if self.use_global and self.use_local:
            assert num_heads % 2 == 0, "Heads must be even to split into Local/Global branches equally."

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        
        _, Hp, Wp, _ = x.shape

        qkv = self.qkv(x) 
        qkv = qkv.reshape(B, Hp, Wp, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        # Choose split point for head allocation.
        if self.use_local and self.use_global:
            # Default: half local, half global.
            split_idx = self.num_heads // 2
        elif self.use_local and not self.use_global:
            # Local-only: assign all heads to local.
            split_idx = self.num_heads 
        elif not self.use_local and self.use_global:
            # Global-only: assign all heads to global.
            split_idx = 0
        else:
            # Fallback (should be prevented by init assertions).
            split_idx = self.num_heads // 2

        out_local = None
        out_global = None

        # --- Branch A: Local (Window Attention) ---
        if self.use_local:
            # Local heads: [:split_idx]
            q_local = q[:, :split_idx].permute(0, 2, 3, 1, 4).reshape(B, Hp, Wp, -1)
            k_local = k[:, :split_idx].permute(0, 2, 3, 1, 4).reshape(B, Hp, Wp, -1)
            v_local = v[:, :split_idx].permute(0, 2, 3, 1, 4).reshape(B, Hp, Wp, -1)
            
            # Active head count (for reshaping).
            curr_heads = split_idx 

            q_local_win = window_partition(q_local, self.window_size).view(-1, self.window_size*self.window_size, curr_heads, self.head_dim).transpose(1, 2)
            k_local_win = window_partition(k_local, self.window_size).view(-1, self.window_size*self.window_size, curr_heads, self.head_dim).transpose(1, 2)
            v_local_win = window_partition(v_local, self.window_size).view(-1, self.window_size*self.window_size, curr_heads, self.head_dim).transpose(1, 2)
            
            a_local = F.scaled_dot_product_attention(
                q_local_win, k_local_win, v_local_win,
                dropout_p=self.attn_drop.p if self.training else 0.,
                scale=self.scale
            )
            
            a_local = a_local.transpose(1, 2).reshape(-1, self.window_size*self.window_size, curr_heads*self.head_dim)
            out_local = window_reverse(a_local, self.window_size, Hp, Wp) 

        # --- Branch B: Global (Grid Attention) ---
        if self.use_global:
            # Global heads: [split_idx:]
            q_global = q[:, split_idx:].permute(0, 2, 3, 1, 4).reshape(B, Hp, Wp, -1)
            k_global = k[:, split_idx:].permute(0, 2, 3, 1, 4).reshape(B, Hp, Wp, -1)
            v_global = v[:, split_idx:].permute(0, 2, 3, 1, 4).reshape(B, Hp, Wp, -1)
            
            # Active head count.
            curr_heads = self.num_heads - split_idx

            grid_seq_len = (Hp // self.window_size) * (Wp // self.window_size)
            q_global_grid = grid_partition(q_global, self.window_size).view(-1, grid_seq_len, curr_heads, self.head_dim).transpose(1, 2)
            k_global_grid = grid_partition(k_global, self.window_size).view(-1, grid_seq_len, curr_heads, self.head_dim).transpose(1, 2)
            v_global_grid = grid_partition(v_global, self.window_size).view(-1, grid_seq_len, curr_heads, self.head_dim).transpose(1, 2)

            a_global = F.scaled_dot_product_attention(
                q_global_grid, k_global_grid, v_global_grid,
                dropout_p=self.attn_drop.p if self.training else 0.,
                scale=self.scale
            )

            a_global = a_global.transpose(1, 2).reshape(-1, grid_seq_len, curr_heads*self.head_dim)
            out_global = grid_reverse(a_global, self.window_size, Hp, Wp) 

        # Merge local/global outputs.
        if self.use_local and self.use_global:
            x = torch.cat([out_local, out_global], dim=-1)
        elif self.use_local:
            x = out_local
        elif self.use_global:
            x = out_global
        
        # Remove padding.
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]
            
        x = x.reshape(B, N, C) 
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LocalTurbulenceBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, **kwargs):
        super(LocalTurbulenceBlock, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.spatial_mixing = nn.Conv2d(dim, dim, 5, padding=2, groups=dim) 
        self.drop_path = SimpleDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.GroupNorm(8, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFeatureMixer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x_in):
        pos = self.pos_embed(x_in)
        x_pos = x_in + pos 
        x_block0_res = x_pos
        x_n1 = self.norm1(x_pos)
        x_c1 = self.conv1(x_n1)
        x_mix = self.spatial_mixing(x_c1)
        x_c2 = self.conv2(x_mix)
        x_dropped1 = self.drop_path(x_c2)
        x_block1 = x_block0_res + x_dropped1
        x_block1_res = x_block1
        x_n2 = self.norm2(x_block1)
        x_mlp_out = self.mlp(x_n2)
        x_dropped2 = self.drop_path(x_mlp_out)
        x_out = x_block1_res + x_dropped2
        return x_out

class GlobalCirculationBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        init_value=1e-6,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        # Pass ablation toggles through the stack.
        use_global=True,
        use_local=True
    ):
        super(GlobalCirculationBlock, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        
        self.attn = DualScaleOceanAttention(
            dim,
            num_heads=num_heads,
            window_size=8, 
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            # Propagate toggles.
            use_global=use_global,
            use_local=use_local
        )
        
        self.drop_path = SimpleDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeatureMixer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape 
        x = x.flatten(2).transpose(1, 2) 
        
        shortcut = x.clone()
        attn_out = self.attn(self.norm1(x), H, W)
        x = shortcut + self.drop_path(self.layer_scale_1 * attn_out)
        
        shortcut = x.clone()
        mlp_out = self.mlp(self.norm2(x))
        x = shortcut + self.drop_path(self.layer_scale_2 * mlp_out)
        
        x = x.transpose(1, 2).reshape(B, N, H, W) 
        return x

def PhysicsInformedBlockSelector(embed_dims, mlp_ratio=4., drop=0., drop_path=0., init_value=1e-6, block_type='Conv', use_global=True, use_local=True):
    if block_type == 'Conv':
        # LocalTurbulenceBlock is convolution-only.
        return LocalTurbulenceBlock(dim=embed_dims, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    else:
        # GlobalCirculationBlock consumes the branch toggles.
        return GlobalCirculationBlock(
            dim=embed_dims, num_heads=4, mlp_ratio=mlp_ratio, qkv_bias=False, 
            drop=drop, drop_path=drop_path, init_value=init_value,
            use_global=use_global, use_local=use_local
        )

class DynamicsEvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0, use_global=True, use_local=True):
        super(DynamicsEvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
        
        self.block = PhysicsInformedBlockSelector(
            in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, 
            block_type=block_type,
            use_global=use_global, use_local=use_local
        )
        
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        if self.in_channels != self.out_channels:
            z = self.reduction(z)
        return z

class LatentEvolutionCore(nn.Module):
    def __init__(self, channel_in, channel_hid, num_layers, mlp_ratio=4., drop=0.0, drop_path=0.1, use_global=True, use_local=True):
        super(LatentEvolutionCore, self).__init__()
        self.num_layers = num_layers
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, num_layers)]
        
        layers = [DynamicsEvolutionLayer(channel_in, channel_hid, mlp_ratio, drop, dpr[0], layer_i=0, use_global=use_global, use_local=use_local)]
        for i in range(1, num_layers - 1):
            layers.append(DynamicsEvolutionLayer(channel_hid, channel_hid, mlp_ratio, drop, dpr[i], layer_i=i, use_global=use_global, use_local=use_local))
        layers.append(DynamicsEvolutionLayer(channel_hid, channel_in, mlp_ratio, drop, drop_path, layer_i=num_layers - 1, use_global=use_global, use_local=use_local))
        
        self.evolution_operator = nn.Sequential(*layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        z = x.view(B * T, C, H, W)
        for i in range(self.num_layers):
            z = self.evolution_operator[i](z)
        _, C_new, H_new, W_new = z.shape
        z = z.view(B, T, C_new, H_new, W_new)
        return z

# --- Encoder / decoder ---

def generate_grid_strides(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]

class OceanDownsampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(OceanDownsampler, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=stride // 2)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=False)
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm: y = self.act(self.norm(y))
        return y

class RestrictionBlock(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(RestrictionBlock, self).__init__()
        if stride == 1: transpose = False
        self.op = OceanDownsampler(C_in, C_out, kernel_size=3, stride=stride, padding=1, transpose=transpose, act_norm=act_norm)
    def forward(self, x): return self.op(x)

class StateEncoder(nn.Module):
    def __init__(self, C_in, spatial_hidden_dim, num_spatial_layers):
        super(StateEncoder, self).__init__()
        strides = generate_grid_strides(num_spatial_layers)
        self.enc = nn.Sequential(
            RestrictionBlock(C_in, spatial_hidden_dim, stride=strides[0]), 
            *[RestrictionBlock(spatial_hidden_dim, spatial_hidden_dim, stride=s) for s in strides[1:]]
        )
    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)): latent = self.enc[i](latent)
        return latent, enc1

class StateDecoder(nn.Module):
    def __init__(self, spatial_hidden_dim, C_out, num_spatial_layers):
        super(StateDecoder, self).__init__()
        strides = generate_grid_strides(num_spatial_layers, reverse=True)
        self.dec = nn.Sequential(
            *[RestrictionBlock(spatial_hidden_dim, spatial_hidden_dim, stride=s, transpose=True) for s in strides[:-1]], 
            RestrictionBlock(2 * spatial_hidden_dim, spatial_hidden_dim, stride=strides[-1], transpose=True)
        )
        self.state_reconstruction = nn.Conv2d(spatial_hidden_dim, C_out, 1)
        
    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1): hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.state_reconstruction(Y)
        return Y

# --- Main Model ---

class OceanDynamicsModel(nn.Module):
    def __init__(self, shape_in, embed_dim=256, output_channels=93, latent_dim=512, num_spatial_layers=4, num_temporal_layers=8,
                 Global_Branch=True, Local_Branch=True):  # top-level API toggles
        super(OceanDynamicsModel, self).__init__()
        T, C, H, W = shape_in
        self.latent_H = int(H / 2 ** (num_spatial_layers / 2)) + 1 if H % 3 == 0 else int(H / 2 ** (num_spatial_layers / 2))
        self.latent_W = int(W / 2 ** (num_spatial_layers / 2))
        
        self.state_encoder = StateEncoder(C, embed_dim, num_spatial_layers)
        
        self.dynamics_core = LatentEvolutionCore(
            channel_in=embed_dim, 
            channel_hid=latent_dim, 
            num_layers=num_temporal_layers, 
            mlp_ratio=8.0, 
            drop_path=0.1,
            # Propagate toggles.
            use_global=Global_Branch,
            use_local=Local_Branch
        )
        
        self.state_decoder = StateDecoder(embed_dim, output_channels, num_spatial_layers)

    def forward(self, input_state):
        batch_size, temporal_length, channels, height, width = input_state.shape
        reshaped_input = input_state.view(batch_size * temporal_length, channels, height, width)
        
        encoded_features, skip_connection = self.state_encoder(reshaped_input)
        
        _, encoded_channels, encoded_height, encoded_width = encoded_features.shape
        encoded_features = encoded_features.view(batch_size, temporal_length, encoded_channels, encoded_height, encoded_width)
        
        temporal_bias = encoded_features
        temporal_hidden = self.dynamics_core(temporal_bias)
        
        reshaped_hidden = temporal_hidden.view(batch_size * temporal_length, encoded_channels, encoded_height, encoded_width)
        decoded_output = self.state_decoder(reshaped_hidden, skip_connection)
        
        final_output = decoded_output.view(batch_size, temporal_length, -1, height, width)
        return final_output

# --- Flux Model for Regional simulation ---

class FluxGatingUnit(nn.Module):
    def __init__(self, channels=12, atmos_channels=3, embed_dim=128, latent_dim=256, 
                 num_spatial_layers=4, num_temporal_layers=4, 
                 mlp_ratio=4., drop=0., drop_path=0.1,
                 Global_Branch=True, Local_Branch=True):  # top-level API toggles
        super(FluxGatingUnit, self).__init__()
        
        combined_channels = 2 * channels 
        
        selector_in_channels = (combined_channels * 3) + (atmos_channels * 2)
        
        self.selector_net = nn.Sequential(
            nn.Conv2d(selector_in_channels, combined_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, combined_channels * 2), 
            nn.GELU(),
            nn.Conv2d(combined_channels * 2, combined_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(combined_channels, 6, kernel_size=1) 
        )
        
        encoder_in_channels = combined_channels + (atmos_channels * 2)
        
        self.encoder = StateEncoder(
            C_in=encoder_in_channels, 
            spatial_hidden_dim=embed_dim, 
            num_spatial_layers=num_spatial_layers
        )
        
        self.core = LatentEvolutionCore(
            channel_in=embed_dim, 
            channel_hid=latent_dim, 
            num_layers=num_temporal_layers, 
            mlp_ratio=mlp_ratio, 
            drop=drop, 
            drop_path=drop_path,
            # Propagate toggles.
            use_global=Global_Branch,
            use_local=Local_Branch
        )
        
        self.decoder = StateDecoder(
            spatial_hidden_dim=embed_dim, 
            C_out=combined_channels, 
            num_spatial_layers=num_spatial_layers
        )

    def forward(self, 
                f_flux_T, c_flux_T_curr, c_flux_T_next,
                f_flux_S, c_flux_S_curr, c_flux_S_next,
                c_A_curr, c_A_next): 
        
        ocean_fluxes = torch.cat([
            f_flux_T, f_flux_S, 
            c_flux_T_curr, c_flux_S_curr, 
            c_flux_T_next, c_flux_S_next
        ], dim=1)
        
        atmos_forcing = torch.cat([c_A_curr, c_A_next], dim=1)
        
        selector_input = torch.cat([ocean_fluxes, atmos_forcing], dim=1)
        
        raw_weights = self.selector_net(selector_input)
        weights_T = F.softmax(raw_weights[:, 0:3, :, :], dim=1)
        weights_S = F.softmax(raw_weights[:, 3:6, :, :], dim=1)
        
        w_f_t, w_c_t, w_n_t = weights_T[:, 0:1], weights_T[:, 1:2], weights_T[:, 2:3]
        w_f_s, w_c_s, w_n_s = weights_S[:, 0:1], weights_S[:, 1:2], weights_S[:, 2:3]
        
        pre_fused_T = w_f_t * f_flux_T + w_c_t * c_flux_T_curr + w_n_t * c_flux_T_next
        pre_fused_S = w_f_s * f_flux_S + w_c_s * c_flux_S_curr + w_n_s * c_flux_S_next
        
        x_in = torch.cat([pre_fused_T, pre_fused_S, atmos_forcing], dim=1)
        
        enc_feat, skip = self.encoder(x_in)
        
        B, C_enc, H_enc, W_enc = enc_feat.shape
        enc_feat_5d = enc_feat.view(B, 1, C_enc, H_enc, W_enc)
        
        hid_5d = self.core(enc_feat_5d)
        
        hid = hid_5d.view(B, C_enc, H_enc, W_enc)
        
        correction = self.decoder(hid, skip)
        
        C = f_flux_T.shape[1]
        corr_T, corr_S = torch.split(correction, C, dim=1)
        
        final_flux_T = f_flux_T + corr_T
        final_flux_S = f_flux_S + corr_S
        
        return final_flux_T, final_flux_S
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    inputs = torch.randn(1, 2, 97, 360, 720) 
    print("Initializing OceanDynamicsModel with Ablation Control...")
    
    # Example: local branch only.
    model = OceanDynamicsModel(
        shape_in=(2, 97, 360, 720), 
        embed_dim=256, 
        output_channels=93, 
        latent_dim=512, 
        num_spatial_layers=4, 
        num_temporal_layers=8,
        Global_Branch=False,  # disable global branch
        Local_Branch=True     # enable local branch
    )
    print(f"Model Parameters (Local Only): {count_parameters(model) / 1e6:.2f} M")
    
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        output = model(inputs)
    else:
        output = model(inputs)
    print("Output Shape:", output.shape)
