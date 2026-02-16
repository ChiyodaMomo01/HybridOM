import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import netCDF4 as nc

def create_hom_parameters(mask_path, mask_ori_path, cmems_path, res, device, R, Omega, t_mask_s, t_mask_e):
    """
    Create grid parameters including boundary mask, lat/lon coordinates,
    grid spacing, layer thicknesses, and Coriolis parameter.
    """
    # Load boundary mask (Specific regional slicing and concatenation)
    # Slicing corresponds to specific region requirements (e.g., crossing meridian)
    mask_1 = np.load(mask_path)[0:23:2, 420:600, 1200:1440]
    mask_2 = np.load(mask_path)[0:23:2, 420:600, 0:120]
    mask = np.concatenate([mask_1, mask_2], axis=2)
    
    mask_ori_1 = np.load(mask_ori_path)[0:23:2, 420:600, 1200:1440]
    mask_ori_2 = np.load(mask_ori_path)[0:23:2, 420:600, 0:120]
    mask_ori = np.concatenate([mask_ori_1, mask_ori_2], axis=2)
    
    _, H, W = mask.shape
    t_mask = mask.copy()
    
    # Note: Specific pole/equator masking logic removed based on active code configuration
    
    # Create latitude and longitude grids
    lats = np.linspace(15 + 1/4, 60 - 1/4, H)
    lons = np.linspace(0 + 1/4, 90 - 1/4, W)
    
    # Calculate grid spacing (meters)
    dy = torch.full((H, W), np.pi * R / (180/(1/4)), dtype=torch.float32, device=device)
    lats_tensor = torch.tensor(lats, dtype=torch.float32, device=device)
    dx = dy * torch.cos(torch.deg2rad(lats_tensor))[:, None] * 2
    
    # Load and process vertical layer thicknesses
    data = nc.Dataset(cmems_path)
    
    # Select specific depth levels
    depth_indices = [0, 4, 8, 12, 16, 20, 22, 24, 26, 28, 30, 32]
    depth_data = data.variables['depth'][depth_indices]
    
    if hasattr(depth_data, 'mask'):
        depth_array = depth_data.filled(np.nan)
    else:
        depth_array = np.array(depth_data, dtype=np.float32)
    
    z_cor = torch.tensor(depth_array, dtype=torch.float32, device=device)
    
    def compute_dz(z_cor):
        """
        Calculate layer thickness (dz) for Finite Volume Method.
        Input: z_cor (N, 1) Tensor, grid center coordinates from surface to bottom.
        Output: dz (N, 1) Tensor.
        """
        N = z_cor.shape[0]
        z_b = torch.zeros(N + 1, 1, dtype=z_cor.dtype, device=z_cor.device)
        z_b[0] = 0.0  # Surface boundary
        
        for k in range(1, N + 1):
            z_b[k] = 2 * z_cor[k - 1] - z_b[k - 1]
        
        dz = z_b[1:] - z_b[:-1]
        return dz
    
    dz = compute_dz(z_cor)
    
    # Calculate Coriolis parameter (s^-1)
    f = 2 * Omega * torch.sin(torch.deg2rad(lats_tensor)).unsqueeze(1).expand(-1, W)
    
    # Move tensors to device
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
    mask_ori_tensor = torch.tensor(mask_ori, dtype=torch.float32, device=device)
    t_mask_tensor = torch.tensor(t_mask, dtype=torch.float32, device=device)
    lats_tensor_2d = torch.tensor(lats, dtype=torch.float32, device=device).unsqueeze(1).expand(-1, W)
    lons_tensor_2d = torch.tensor(lons, dtype=torch.float32, device=device).unsqueeze(0).expand(H, -1)
    
    params = {
        'mask': mask_tensor,
        'mask_ori': mask_ori_tensor,
        't_mask': t_mask_tensor,
        'lats': lats_tensor_2d,
        'lons': lons_tensor_2d,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'f': f,
        'z_cor': z_cor
    }
    
    return params

class DynamicalCore(nn.Module):
    def __init__(self, D, H, W, dx, dy, dz, z_cor, mask, mask_ori, t_mask, f, 
                 rho_0=1026.0, g=9.81, epsilon=0.1, dt=7200.0, num_step=12, 
                 ah=500.0, kh=100.0, use_checkpoint=False):
        """
        Args:
            ah (float): Horizontal viscosity for momentum/vorticity (m^2/s). 
            kh (float): Horizontal diffusivity for tracers T/S (m^2/s). 
        """
        super(DynamicalCore, self).__init__()
        
        self.D, self.H, self.W = D, H, W
        self.num_step = num_step
        self.use_checkpoint = use_checkpoint
        
        # --- Register Grid Buffers ---
        self.register_buffer('dx', dx.view(1, 1, H, W))
        self.register_buffer('dy', dy.view(1, 1, H, W))
        self.register_buffer('dz', dz.view(1, D, 1, 1))
        self.register_buffer('area', (dx * dy).view(1, 1, H, W)) 
        
        self.register_buffer('z_cor', z_cor.view(1, D, 1, 1))
        self.register_buffer('mask', mask.view(1, D, H, W))      # Tracer mask
        self.register_buffer('t_mask', t_mask.view(1, D, H, W))  # Vorticity mask
        self.register_buffer('f', f.view(1, 1, H, W))
        
        # --- Constants ---
        self.register_buffer('rho_0', torch.tensor(rho_0))
        self.register_buffer('g', torch.tensor(g))
        self.register_buffer('epsilon', torch.tensor(epsilon))
        self.register_buffer('dt', torch.tensor(dt))
        
        # Diffusion/Viscosity Coefficients
        self.register_buffer('ah', torch.tensor(ah)) # For Vorticity
        self.register_buffer('kh', torch.tensor(kh)) # For Tracers (T, S)
        
        # EOS Constants
        eos_params = {
            'a0': 1.6550e-1, 'b0': 7.6554e-1, 
            'lambda1': 5.9520e-2, 'lambda2': 5.4914e-4,
            'nu': 2.4341e-3, 'mu1': 1.4970e-4, 'mu2': 1.1090e-5
        }
        for name, val in eos_params.items():
            self.register_buffer(name, torch.tensor(val))

        # Pre-compute depth levels for EOS
        cum_dz = torch.cumsum(dz.view(1, D), dim=1)
        z_levels = cum_dz - 0.5 * dz.view(1, D)
        self.register_buffer('z_levels', z_levels.view(1, D, 1, 1))
        
        # --- Smoothing Kernel for Tendencies ---
        # Create a mild 3x3 smoothing kernel
        # Weight structure: Center weighted high, edges low
        # [[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16.0
        smoothing_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
        # Expand dims for F.conv2d (Out, In/Groups, H, W)
        # Using groups=channels for depthwise separable convolution
        self.register_buffer('smooth_kernel', smoothing_kernel.view(1, 1, 3, 3))

    # =========================================================================
    #  FVM Core Operators (Flux & Divergence)
    # =========================================================================

    def compute_flux_x(self, phi, u):
        """Calculate zonal advection flux: Flux = u * phi_face (Upwind/Reconstruction)"""
        phi_center = phi
        phi_east = torch.roll(phi, shifts=-1, dims=-1) # i+1
        
        # Upwind scheme
        mask_pos = (u > 0).float()
        phi_at_face = mask_pos * phi_center + (1.0 - mask_pos) * phi_east
        
        return u * phi_at_face

    def compute_flux_y(self, phi, v):
        """Calculate meridional advection flux"""
        phi_center = phi
        phi_north = torch.roll(phi, shifts=-1, dims=-2) # j+1
        
        mask_pos = (v > 0).float()
        phi_at_face = mask_pos * phi_center + (1.0 - mask_pos) * phi_north
        
        return v * phi_at_face

    def compute_flux_z(self, phi, w):
        """Calculate vertical advection flux"""
        phi_padded = F.pad(phi, (0,0, 0,0, 0,1), mode='replicate') 
        phi_upper = phi_padded[:, :-1, :, :] # k
        phi_lower = phi_padded[:, 1:, :, :]  # k+1
        
        mask_up = (w > 0).float()
        phi_at_face = mask_up * phi_lower + (1.0 - mask_up) * phi_upper
        
        return w * phi_at_face

    def flux_divergence(self, flux_x, flux_y, flux_z=None):
        """Calculate flux divergence: Div = (F_out - F_in) / dx"""
        # X direction
        flux_x_west = torch.roll(flux_x, shifts=1, dims=-1)
        div_x = (flux_x - flux_x_west) / self.dx
        
        # Y direction
        flux_y_south = torch.roll(flux_y, shifts=1, dims=-2)
        div_y = (flux_y - flux_y_south) / self.dy
        
        div = div_x + div_y
        
        if flux_z is not None:
            # Z direction
            zeros = torch.zeros_like(flux_z[:, :1])
            flux_z_shifted = torch.cat([zeros, flux_z[:, :-1]], dim=1)
            div_z = (flux_z - flux_z_shifted) / self.dz
            div = div + div_z
            
        return div

    # =========================================================================
    #  Diffusion Operators (Laplacian with Boundary Masking)
    # =========================================================================
    
    def compute_laplacian_flux(self, phi, coeff, mask_type='tracer'):
        """
        Calculate diffusion flux F = -K * grad(phi).
        Critical: Applies interface masking. If Cell(i) is water and Cell(i+1) is land,
        flux at interface i+1/2 must be 0.
        """
        mask = self.mask if mask_type == 'tracer' else self.t_mask
        
        # --- X Direction ---
        # Grad defined at east face i+1/2
        phi_east = torch.roll(phi, shifts=-1, dims=-1)
        grad_x = (phi_east - phi) / self.dx
        
        # Interface mask X
        mask_east = torch.roll(mask, shifts=-1, dims=-1)
        interface_mask_x = mask * mask_east
        
        flux_diff_x = -coeff * grad_x * interface_mask_x 
        
        # --- Y Direction ---
        # Grad defined at north face j+1/2
        phi_north = torch.roll(phi, shifts=-1, dims=-2)
        grad_y = (phi_north - phi) / self.dy
        
        # Interface mask Y
        mask_north = torch.roll(mask, shifts=-1, dims=-2)
        interface_mask_y = mask * mask_north
        
        flux_diff_y = -coeff * grad_y * interface_mask_y
        
        return flux_diff_x, flux_diff_y

    # =========================================================================
    #  Public API: Flux & Divergence
    # =========================================================================

    def compute_flux(self, u, v, T, S):
        """
        Calculate total flux (Advection + Diffusion) for T and S.
        Returns concatenated (x, y) components.
        """
        # --- Temperature ---
        adv_flux_T_x = self.compute_flux_x(T, u)
        adv_flux_T_y = self.compute_flux_y(T, v)
        diff_flux_T_x, diff_flux_T_y = self.compute_laplacian_flux(T, self.kh, mask_type='tracer')
        
        total_flux_T_x = adv_flux_T_x + diff_flux_T_x
        total_flux_T_y = adv_flux_T_y + diff_flux_T_y
        
        # --- Salinity ---
        adv_flux_S_x = self.compute_flux_x(S, u)
        adv_flux_S_y = self.compute_flux_y(S, v)
        diff_flux_S_x, diff_flux_S_y = self.compute_laplacian_flux(S, self.kh, mask_type='tracer')
        
        total_flux_S_x = adv_flux_S_x + diff_flux_S_x
        total_flux_S_y = adv_flux_S_y + diff_flux_S_y
        
        # Concatenate x and y components -> [B, 2*D, H, W]
        flux_T = torch.cat([total_flux_T_x, total_flux_T_y], dim=1)
        flux_S = torch.cat([total_flux_S_x, total_flux_S_y], dim=1)
        
        return flux_T, flux_S

    def compute_div(self, flux_combined):
        """
        Calculate divergence of the combined flux tensor.
        Args: flux_combined [B, 2*D, H, W]
        """
        D = flux_combined.shape[1] // 2
        flux_x, flux_y = torch.split(flux_combined, D, dim=1)
        
        div = self.flux_divergence(flux_x, flux_y)
        return div * self.mask
    
    # =========================================================================
    #  Physics Calculations (EOS, Geostrophy, etc.)
    # =========================================================================

    def grad_central(self, phi, spacing, dim):
        if dim == 'x':
            phi_l = torch.roll(phi, shifts=1, dims=-1)
            phi_r = torch.roll(phi, shifts=-1, dims=-1)
            return (phi_r - phi_l) / (2.0 * spacing)
        elif dim == 'y':
            phi_d = torch.roll(phi, shifts=1, dims=-2)
            phi_u = torch.roll(phi, shifts=-1, dims=-2)
            return (phi_u - phi_d) / (2.0 * spacing)
        return None

    def compute_rho(self, S_ano, T_ano, S_mean, T_mean):
        """
        Calculate density anomaly via EOS.
        Reconstruct total T/S, calculate rho, then subtract rho_0.
        """
        T_total = T_ano + T_mean
        S_total = S_ano + S_mean
        
        dT = T_total - 10.0
        dS = S_total - 35.0
        
        inner_T = 1 + 0.5 * self.lambda1 * dT + self.mu1 * self.z_levels
        inner_S = 1 - 0.5 * self.lambda2 * dS - self.mu2 * self.z_levels
        
        # EOS formula
        da = (-self.a0 * inner_T * dT + self.b0 * inner_S * dS - self.nu * dT * dS) / self.rho_0
        rho_prime = self.rho_0 * da
        
        return rho_prime

    def compute_pressure(self, rho_prime, eta_ano, eta_mean):
        """
        Calculate hydrostatic pressure.
        P(z) = g * rho_0 * eta + integral(g * rho' dz)
        """
        eta_total = eta_ano + eta_mean
        
        # Barotropic term (Surface)
        p_surface = self.g * self.rho_0 * eta_total.unsqueeze(1)
        
        # Baroclinic term
        layer_contrib = 0.5 * self.g * rho_prime * self.dz
        cumsum_contrib = torch.cumsum(layer_contrib, dim=1)
        
        zeros = torch.zeros_like(layer_contrib[:, :1, :, :])
        sum_prev = torch.cat([zeros, cumsum_contrib[:, :-1, :, :]], dim=1)
        
        p_hydrostatic = p_surface + 2 * sum_prev + layer_contrib
        
        return p_hydrostatic

    def compute_w(self, u, v):
        """Calculate vertical velocity w based on continuity equation."""
        u_west = torch.roll(u, shifts=1, dims=-1)
        div_x = (u - u_west) / self.dx
        
        v_south = torch.roll(v, shifts=1, dims=-2)
        div_y = (v - v_south) / self.dy
        
        div_h = div_x + div_y
        term = -div_h * self.dz
        w = torch.cumsum(term, dim=1)
        
        return w

    def compute_vort(self, u, v):
        """Calculate relative vorticity."""
        return self.grad_central(v, self.dx, 'x') - self.grad_central(u, self.dy, 'y')

    def compute_geostrophic_velocity(self, p):
        """
        Calculate Geostrophic Velocity on a C-grid.
        u_g = - (1 / (rho0 * f)) * dp/dy
        v_g = + (1 / (rho0 * f)) * dp/dx
        
        Requires interpolation of pressure gradients to velocity points.
        """
        # 1. Raw gradients at faces
        # dp/dx @ u-face (i+1/2, j)
        p_east = torch.roll(p, shifts=-1, dims=-1)
        dp_dx = (p_east - p) / self.dx 
        
        # dp/dy @ v-face (i, j+1/2)
        p_north = torch.roll(p, shifts=-1, dims=-2)
        dp_dy = (p_north - p) / self.dy
        
        # 2. Cross-interpolation
        # Interpolate dp/dy (v-face) -> u-face
        dp_dy_center = 0.5 * (dp_dy + torch.roll(dp_dy, shifts=1, dims=-2))
        dp_dy_at_u   = 0.5 * (dp_dy_center + torch.roll(dp_dy_center, shifts=-1, dims=-1))
        
        # Interpolate dp/dx (u-face) -> v-face
        dp_dx_center = 0.5 * (dp_dx + torch.roll(dp_dx, shifts=1, dims=-1))
        dp_dx_at_v   = 0.5 * (dp_dx_center + torch.roll(dp_dx_center, shifts=-1, dims=-2))
        
        # 3. Interpolate Coriolis f
        f_u = 0.5 * (self.f + torch.roll(self.f, shifts=-1, dims=-1))
        f_v = 0.5 * (self.f + torch.roll(self.f, shifts=-1, dims=-2))

        # 4. Calculate velocities
        u_g = - (1.0 / (self.rho_0 * f_u)) * dp_dy_at_u * self.t_mask
        v_g =   (1.0 / (self.rho_0 * f_v)) * dp_dx_at_v * self.t_mask
        
        return u_g, v_g
    
    def apply_smoothing(self, field):
        """
        Apply 2D horizontal smoothing to input 4D field (B, D, H, W).
        Uses replicate padding at boundaries.
        """
        B, D, H, W = field.shape
        
        # 1. Adjust Kernel for input channel D
        # Use groups=D for depthwise convolution
        weight = self.smooth_kernel.repeat(D, 1, 1, 1)
        
        # 2. Padding (maintain spatial dimensions)
        field_padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
        
        # 3. Convolution
        field_smoothed = F.conv2d(field_padded, weight, groups=D)
        
        return field_smoothed

    def compute_tendencies(self, u, v, T, S, eta, vort, sic, sit, usi, vsi, 
                           T_mean, S_mean, eta_mean):
        """
        Calculate physical tendencies.
        1. Internal calculation of rho, pressure, and geostrophic velocities.
        2. Vorticity advection uses geostrophic velocities (ug, vg).
        3. T/S advection uses input velocities (u, v).
        """
        
        # 1. Vertical velocity
        w = self.compute_w(u, v)
        
        # 2. Diagnostics
        rho_prime = self.compute_rho(S, T, S_mean, T_mean)
        p_hydro = self.compute_pressure(rho_prime, eta, eta_mean)
        ug, vg = self.compute_geostrophic_velocity(p_hydro)

        # 3. Advection
        # T/S use u, v
        flux_T_x = self.compute_flux_x(T, u)
        flux_T_y = self.compute_flux_y(T, v)
        flux_S_x = self.compute_flux_x(S, u)
        flux_S_y = self.compute_flux_y(S, v)
        
        # Vorticity uses ug, vg
        flux_vort_x = self.compute_flux_x(vort, ug) # Note: Uses geostrophic u
        flux_vort_y = self.compute_flux_y(vort, vg) # Note: Uses geostrophic v

        # Divergence
        adv_T = -self.flux_divergence(flux_T_x, flux_T_y) * self.mask
        adv_S = -self.flux_divergence(flux_S_x, flux_S_y) * self.mask
        adv_vort = -self.flux_divergence(flux_vort_x, flux_vort_y) * self.mask

        # 4. Diffusion/Dissipation
        d_flux_T_x, d_flux_T_y = self.compute_laplacian_flux(T, self.kh, mask_type='tracer')
        d_flux_S_x, d_flux_S_y = self.compute_laplacian_flux(S, self.kh, mask_type='tracer')
        d_flux_vort_x, d_flux_vort_y = self.compute_laplacian_flux(vort, self.ah, mask_type='vort')

        diff_T = -self.flux_divergence(d_flux_T_x, d_flux_T_y) * self.mask
        diff_S = -self.flux_divergence(d_flux_S_x, d_flux_S_y) * self.mask
        diff_vort = -self.flux_divergence(d_flux_vort_x, d_flux_vort_y) * self.t_mask
        
        total_T_tendency = adv_T + diff_T
        total_S_tendency = adv_S + diff_S
        total_vort_tendency = adv_vort + diff_vort

        # --- Apply Mild Horizontal Smoothing to Tendencies ---
        # Smooth only T, S, Vort tendencies to suppress numerical noise
        total_T_tendency = self.apply_smoothing(total_T_tendency) * self.mask
        total_S_tendency = self.apply_smoothing(total_S_tendency) * self.mask
        total_vort_tendency = self.apply_smoothing(total_vort_tendency) * self.t_mask

        # 5. Sea Ice Transport
        mask_surf = self.mask[:, 0:1]
        usi_3d = usi.unsqueeze(1)
        vsi_3d = vsi.unsqueeze(1)
        sic_3d = sic.unsqueeze(1)
        sit_3d = sit.unsqueeze(1)
        
        f_sic_x = self.compute_flux_x(sic_3d, usi_3d)
        f_sic_y = self.compute_flux_y(sic_3d, vsi_3d)
        f_sit_x = self.compute_flux_x(sit_3d, usi_3d)
        f_sit_y = self.compute_flux_y(sit_3d, vsi_3d)
        
        F_sic = -self.flux_divergence(f_sic_x, f_sic_y).squeeze(1) * mask_surf.squeeze(1)
        F_sit = -self.flux_divergence(f_sit_x, f_sit_y).squeeze(1) * mask_surf.squeeze(1)

        return total_T_tendency, total_S_tendency, total_vort_tendency, F_sic, F_sit

    def _step_forward(self, u, v, T, S, eta, sic, sit, usi, vsi, vort, 
                      prev_F_T, prev_F_S, prev_F_vort, prev_F_sic, prev_F_sit, step_idx_tensor,
                      T_mean, S_mean, eta_mean):
        
        step = int(step_idx_tensor.item())
        
        F_T, F_S, F_vort, F_sic, F_sit = self.compute_tendencies(
            u, v, T, S, eta, vort, sic, sit, usi, vsi, T_mean, S_mean, eta_mean
        )

        # AB2 (Adams-Bashforth) Time Stepping Scheme
        def ab2_step(field, F_cur, F_prev):
            if step == 0:
                return field + self.dt * F_cur
            else:
                return field + self.dt * ((1.5 + self.epsilon) * F_cur - (0.5 + self.epsilon) * F_prev)

        new_T = ab2_step(T, F_T, prev_F_T)
        new_S = ab2_step(S, F_S, prev_F_S)
        new_vort = ab2_step(vort, F_vort, prev_F_vort)
        new_sic = ab2_step(sic, F_sic, prev_F_sic)
        new_sit = ab2_step(sit, F_sit, prev_F_sit)
        
        return new_T, new_S, new_vort, new_sic, new_sit, F_T, F_S, F_vort, F_sic, F_sit

    def forward(self, u, v, T, S, eta, sic, sit, usi, vsi, T_mean, S_mean, eta_mean):
        """
        Forward pass for time integration.
        """
        vort = self.compute_vort(u, v)
        vort_init = vort.clone()

        prev_F_T = torch.zeros_like(T)
        prev_F_S = torch.zeros_like(S)
        prev_F_vort = torch.zeros_like(vort)
        prev_F_sic = torch.zeros_like(sic)
        prev_F_sit = torch.zeros_like(sit)

        current_T, current_S, current_vort = T, S, vort
        current_sic, current_sit = sic, sit
        
        for step in range(self.num_step):
            step_tensor = torch.tensor(step, device=u.device)
            
            if self.use_checkpoint and self.training:
                current_T, current_S, current_vort, current_sic, current_sit, \
                F_T, F_S, F_vort, F_sic, F_sit = checkpoint.checkpoint(
                    self._step_forward,
                    u, v, current_T, current_S, eta, current_sic, current_sit, usi, vsi, current_vort,
                    prev_F_T, prev_F_S, prev_F_vort, prev_F_sic, prev_F_sit, step_tensor,
                    T_mean, S_mean, eta_mean,
                    use_reentrant=False
                )
            else:
                current_T, current_S, current_vort, current_sic, current_sit, \
                F_T, F_S, F_vort, F_sic, F_sit = self._step_forward(
                    u, v, current_T, current_S, eta, current_sic, current_sit, usi, vsi, current_vort,
                    prev_F_T, prev_F_S, prev_F_vort, prev_F_sic, prev_F_sit, step_tensor,
                    T_mean, S_mean, eta_mean
                )
            
            prev_F_T, prev_F_S, prev_F_vort = F_T, F_S, F_vort
            prev_F_sic, prev_F_sit = F_sic, F_sit

        return current_T, current_S, current_vort, vort_init, current_sic, current_sit