import sys
import os
import logging
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data as data_utils
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Add project root to path (portable; avoids hard-coded personal/cluster paths)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Custom Imports
from utils.dynamical_core_regional import DynamicalCore, create_hom_parameters
from dataloader_regional import TestDataset 
# Note: Using dataloader based on your train script (assuming TestDataset is there)
from networks.baseline.baseline_ST.ParamNet import OceanDynamicsModel, FluxGatingUnit

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==============================================================================
# Dynamic Gaussian Smoothing (For Regional Output)
# ==============================================================================
def smooth_regional_data_dynamic(data, max_sigma=0.5, start_day=10):
    """
    Apply time-dependent Gaussian smoothing to a 4D (Time, Channel, Lat, Lon) array.
    Helps visualize long-term rollouts by removing grid-scale noise.
    """
    if data.ndim != 4:
        raise ValueError(f"Input data must be 4D (T, C, H, W). Got shape {data.shape}")
    
    T, C, H, W = data.shape
    output = np.zeros_like(data)
    
    for t in range(T):
        # Ramp up smoothing after start_day
        if t < start_day:
            current_sigma = 0.0
        else:
            denominator = (T - 1 - start_day)
            if denominator <= 0:
                progress = 1.0
            else:
                progress = (t - start_day) / denominator
            current_sigma = max_sigma # Use max_sigma or scale it by progress if desired

        if current_sigma < 1e-6:
            output[t] = data[t]
            continue
            
        pad_width = int(np.ceil(current_sigma * 3))
        pad_axes = ((0, 0), (pad_width, pad_width), (pad_width, pad_width))
        
        data_t = data[t]
        data_padded = np.pad(data_t, pad_axes, mode='edge') # Edge padding for regional
        
        sigma_tuple = (0, current_sigma, current_sigma)
        smoothed_padded = gaussian_filter(data_padded, sigma=sigma_tuple, mode='nearest')
        
        output[t] = smoothed_padded[:, pad_width:-pad_width, pad_width:-pad_width]

    return output

class TimeStepper(nn.Module):
    """
    Full Inference TimeStepper.
    Logic: Physics Flux -> Gating Net -> ILAP Correction -> State Net Correction -> Stabilization
    """
    def __init__(self, model_ilap, model_state, gating_net, dycore, mask_ori, norm_cfg, device):
        super().__init__()
        self.model_ilap = model_ilap
        self.model_state = model_state
        self.gating_net = gating_net
        self.dycore = dycore
        self.register_buffer('mask_ori', mask_ori)
        self.device = device
        
        self.day2sec = 86400.0

        # --- Load Normalization Constants ---
        self.register_buffer('norm_div', torch.tensor(norm_cfg['div'], dtype=torch.float32))
        self.register_buffer('norm_sub', torch.tensor(norm_cfg['sub'], dtype=torch.float32))
        self.register_buffer('norm_div_last', torch.tensor(norm_cfg['div_last'], dtype=torch.float32))

    def _normalize(self, state):
        """Normalization logic for the full state stack"""
        s = state.clone()
        # Division normalization
        s[:,:,0:12]   /= self.norm_div[0]
        s[:,:,12:24]  /= self.norm_div[1]
        s[:,:,24:36]  /= self.norm_div[2]
        s[:,:,36:48]  /= self.norm_div[3]
        s[:,:,48]     /= self.norm_div[4]
        
        s[:,:,55:67]   /= (self.norm_div[0]/4)
        s[:,:,67:79]  /= (self.norm_div[1]/4)
        s[:,:,79:91]  /= (self.norm_div[2]/4)
        s[:,:,91:103]  /= (self.norm_div[3]/4)
        s[:,:,103]     /= (self.norm_div[4]/4)
        
        # Shift and Scale
        indices = [(49, 0), (50, 1), (51, 2), (52, 0), (53, 1), (54, 2)]
        for idx_tensor, idx_const in indices:
            s[:,:,idx_tensor] = (s[:,:,idx_tensor] - self.norm_sub[idx_const]) / self.norm_div_last[idx_const]
        return s

    def _normalize_atmos(self, atmos_tensor):
        """Helper specifically for normalizing standalone Atmos tensors (c_A)."""
        s = atmos_tensor.clone()
        s[:, 0] = (s[:, 0] - self.norm_sub[0]) / self.norm_div_last[0]
        s[:, 1] = (s[:, 1] - self.norm_sub[1]) / self.norm_div_last[1]
        s[:, 2] = (s[:, 2] - self.norm_sub[2]) / self.norm_div_last[2]
        return s

    def _get_damping_factor(self, step_idx, start_val, end_val, start_step=10, end_step=50):
        """Helper to calculate linear ramp damping factor."""
        if step_idx >= end_step:
            return end_val
        else:
            progress = (step_idx - start_step) / (end_step - start_step)
            return start_val - progress * (start_val - end_val)

    def forward(self, 
                # Fine Grid State (t) - X^h_t
                f_T, f_S, f_u, f_v, f_eta, f_A,
                # Coarse Grid State (t) - X^l_t
                c_T_curr, c_S_curr, c_u_curr, c_v_curr, c_eta_curr, c_A_curr,
                # Coarse Grid State (t+1) - X^l_{t+1}
                c_T_next, c_S_next, c_u_next, c_v_next, c_eta_next, c_A_next,
                # Current Step Index
                step_idx=0):
        
        B, D, H, W = f_T.shape
        
        # Placeholders
        zero_ice_2d = torch.zeros((B, H, W), device=self.device)
        zero_ice_3d = torch.zeros((B, 1, H, W), device=self.device)
        zero_mean_T = torch.zeros_like(f_T)
        zero_mean_S = torch.zeros_like(f_S)
        zero_mean_eta = torch.zeros((B, H, W), device=self.device)

        # =====================================================================
        # Step 2: Calculate Flux Terms using dycore API
        # =====================================================================
        f_flux_T, f_flux_S = self.dycore.compute_flux(f_u, f_v, f_T, f_S)
        
        with torch.no_grad():
            c_flux_T_curr, c_flux_S_curr = self.dycore.compute_flux(c_u_curr, c_v_curr, c_T_curr, c_S_curr)
            c_flux_T_next, c_flux_S_next = self.dycore.compute_flux(c_u_next, c_v_next, c_T_next, c_S_next)

        # =====================================================================
        # Step 3: Calculate Fused Flux (F_theta) using Gating Net
        # =====================================================================
        norm_c_A_curr = self._normalize_atmos(c_A_curr)
        norm_c_A_next = self._normalize_atmos(c_A_next)

        fused_flux_T, fused_flux_S = self.gating_net(
            f_flux_T, c_flux_T_curr, c_flux_T_next,
            f_flux_S, c_flux_S_curr, c_flux_S_next,
            norm_c_A_curr, norm_c_A_next 
        )

        scale_factor_state = 0.001 
        intermediate_T = f_T + self.dycore.compute_div(fused_flux_T) * self.day2sec * scale_factor_state
        intermediate_S = f_S + self.dycore.compute_div(fused_flux_S) * self.day2sec * scale_factor_state
        
        # =====================================================================
        # Step 4 (Part A): ILAP Correction (U, V)
        # =====================================================================
        
        # 1. Run Dycore to get Vorticity
        _, _, next_vort_phy, _, _, _ = self.dycore(
            f_u, f_v, f_T, f_S, f_eta,  
            zero_ice_3d, zero_ice_3d, zero_ice_2d, zero_ice_2d,
            zero_mean_T, zero_mean_S, zero_mean_eta 
        )

        # 2. ILAP Input construction
        # f_A usually passed as [B, C, H, W], we need [B, 1, C, H, W] for ILAP input dim 2 concat
        ilap_input = torch.cat([
            next_vort_phy.unsqueeze(1) * 1e4,
            f_u.unsqueeze(1),
            f_v.unsqueeze(1),
            f_A.unsqueeze(1) # Assuming f_A here is the specific slice for this step
        ], dim=2)

        # 3. ILAP Prediction
        next_uv_correction = self.model_ilap(ilap_input)
        
        num_depth = f_u.shape[1]
        delta_u_ilap = next_uv_correction[:, 0, 0:num_depth]
        delta_v_ilap = next_uv_correction[:, 0, num_depth:2*num_depth]
        
        scale_factor_ilap = 0.02
        
        intermediate_u = f_u + delta_u_ilap * scale_factor_ilap
        intermediate_v = f_v + delta_v_ilap * scale_factor_ilap
        intermediate_eta = f_eta 

        # =====================================================================
        # Step 4 (Part B): State Network Correction
        # =====================================================================
        high_res_stack = torch.cat([f_T, f_S, f_u, f_v, f_eta.unsqueeze(1), f_A], dim=1) 
        
        low_res_curr_stack = torch.cat([c_T_curr, c_S_curr, c_u_curr, c_v_curr, c_eta_curr.unsqueeze(1)], dim=1) 
        low_res_next_stack = torch.cat([c_T_next, c_S_next, c_u_next, c_v_next, c_eta_next.unsqueeze(1)], dim=1) 
        low_res_delta_stack = low_res_next_stack - low_res_curr_stack
        
        combined_input = torch.cat([high_res_stack, low_res_delta_stack], dim=1) 
        combined_input = combined_input.unsqueeze(1) 
        norm_combined_input = self._normalize(combined_input)
        
        scale_factor_neural = 0.02 # Matching train script
        S_theta = self.model_state(norm_combined_input) * scale_factor_neural
        
        mask_expanded = self.mask_ori.unsqueeze(0).expand(f_T.shape[0], -1, -1, -1)
        
        # Calculate PROPOSED next state (Before Clamping)
        proposed_T = intermediate_T + S_theta[:, 0, 0:12]
        proposed_S = intermediate_S + S_theta[:, 0, 12:24]
        proposed_u = intermediate_u + S_theta[:, 0, 24:36]
        proposed_v = intermediate_v + S_theta[:, 0, 36:48]
        proposed_eta = intermediate_eta + S_theta[:, 0, 48]

        # ==============================================================================
        # DYNAMIC STABILIZATION: Clamp & Damping (Inference Only Trick)
        # ==============================================================================
        if step_idx >= 11:
            # 1. Calculate Tendencies (Net change per step)
            delta_T = proposed_T - f_T
            delta_S = proposed_S - f_S
            delta_u = proposed_u - f_u
            delta_v = proposed_v - f_v
            
            # 2. Calculate Damping Factors
            damp_temp = self._get_damping_factor(step_idx, start_val=0.95, end_val=0.60)
            damp_salt = self._get_damping_factor(step_idx, start_val=0.95, end_val=0.60)
            damp_vel = self._get_damping_factor(step_idx, start_val=1.00, end_val=0.90)
            
            # 3. Hard Clamping Threshold
            clamp_val_scalar = 1.0
            clamp_val_vel = 1.5
            
            # 4. Apply Stabilization
            delta_T = torch.clamp(delta_T, -clamp_val_scalar, clamp_val_scalar) * damp_temp
            delta_S = torch.clamp(delta_S, -clamp_val_scalar, clamp_val_scalar) * damp_salt
            delta_u = torch.clamp(delta_u, -clamp_val_vel, clamp_val_vel) * damp_vel 
            delta_v = torch.clamp(delta_v, -clamp_val_vel, clamp_val_vel) * damp_vel 
            
            final_T = (f_T + delta_T) * mask_expanded
            final_S = (f_S + delta_S) * mask_expanded
            final_u = (f_u + delta_u) * mask_expanded
            final_v = (f_v + delta_v) * mask_expanded
            final_eta = proposed_eta # ETA is usually stable
            
        else:
            final_T = proposed_T * mask_expanded
            final_S = proposed_S * mask_expanded
            final_u = proposed_u * mask_expanded
            final_v = proposed_v * mask_expanded
            final_eta = proposed_eta

        # Boundaries
        final_T[..., 0] = final_T[..., -1]
        final_S[..., 0] = final_S[..., -1]
        final_u[..., 0] = final_u[..., -1]
        final_v[..., 0] = final_v[..., -1]
        final_eta[..., 0] = final_eta[..., -1]

        return final_T, final_S, final_u, final_v, final_eta


class HybridOMInferrer:
    def __init__(self, config_path='config.yaml'):
        self.load_config(config_path)
        self.setup_ddp()
        self.setup_logging()
        self.setup_data_and_physics()
        self.setup_models()
        
    def load_config(self, path):
        with open(path, 'r') as f:
            self.full_config = yaml.safe_load(f)
        
        self.model_name = self.full_config['selected_model']
        self.model_cfg = self.full_config['models'][self.model_name]
        self.train_cfg = self.full_config['trainings'][self.model_name]
        self.log_cfg = self.full_config['loggings'][self.model_name]
        self.norm_cfg = self.model_cfg['normalization']
        
        set_seed(self.train_cfg['seed'])
        
        self.result_dir = Path(self.log_cfg['result_dir'])
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def setup_ddp(self):
        self.parallel_method = self.train_cfg.get('parallel_method', 'DistributedDataParallel')
        if self.parallel_method == 'DistributedDataParallel':
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.num_gpus = dist.get_world_size()
            self.is_master = (self.local_rank == 0)
        else:
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.num_gpus = torch.cuda.device_count()
            self.is_master = True

    def setup_logging(self):
        if self.is_master:
            logging.basicConfig(
                filename=f"{self.log_cfg['log_dir']}/{self.log_cfg['backbone']}_inference.log",
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            logging.getLogger().addHandler(console)
            logging.info(f"Initialized inference on {self.num_gpus} GPUs.")
        else:
            logging.getLogger().setLevel(logging.WARNING)

    def setup_data_and_physics(self):
        self.params = create_hom_parameters(
            mask_path=self.model_cfg['data_params']['mask_path'],
            mask_ori_path=self.model_cfg['data_params']['mask_ori_path'],
            cmems_path=self.model_cfg['data_params']['cmems_path'],
            res=self.model_cfg['hybridom_params']['res'], 
            device=torch.device('cuda'),
            R=self.model_cfg['hybridom_params']['R'], 
            Omega=self.model_cfg['hybridom_params']['Omega'],
            t_mask_s=self.model_cfg['hybridom_params']['t_mask_s'],
            t_mask_e=self.model_cfg['hybridom_params']['t_mask_e']
        )
        
        # Use TestDataset
        dataset = TestDataset()
        self.sampler = DistributedSampler(dataset, shuffle=False) if self.parallel_method == 'DistributedDataParallel' else None
        
        self.test_loader = data_utils.DataLoader(
            dataset, num_workers=1, batch_size=self.train_cfg['batch_size'], 
            pin_memory=True, shuffle=False, sampler=self.sampler
        )

        self.dycore = DynamicalCore(
            self.model_cfg['hybridom_params']['D'], 
            self.model_cfg['hybridom_params']['H'], 
            self.model_cfg['hybridom_params']['W'],
            self.params['dx'], self.params['dy'], self.params['dz'], 
            self.params['z_cor'], self.params['mask'], self.params['mask_ori'], 
            self.params['t_mask'], self.params['f'], 
            dt=self.model_cfg['hybridom_params']['dt'], 
            num_step=self.model_cfg['hybridom_params']['N_step']
        ).to(self.device)

    def _load_weights(self, model, path):
        if os.path.exists(path):
            if self.is_master:
                logging.info(f"Loading weights from {path}")
            state_dict = torch.load(path, map_location=self.device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=True)
        else:
            if self.is_master:
                logging.error(f"Checkpoint path {path} does not exist!")
            raise FileNotFoundError(f"Checkpoint path {path} does not exist!")

    def setup_models(self):
        model_cls = OceanDynamicsModel 
        
        # 1. Initialize ALL models (ILAP + State + Gating)
        self.model_ilap = model_cls(**self.model_cfg['odm_params_ilap']).to(self.device)
        self.model_state = model_cls(**self.model_cfg['odm_params_state']).to(self.device)
        self.gating_net = FluxGatingUnit(**self.model_cfg['gating_params']).to(self.device)

        ckpt_dir = Path(self.log_cfg['checkpoint_dir'])
        backbone = self.log_cfg['backbone']
        
        # 2. Define Paths
        path_ilap = ckpt_dir / f"{backbone}_ilap_best.pth"
        path_state = ckpt_dir / f"{backbone}_state_best.pth"
        path_gate = ckpt_dir / f"{backbone}_gating_best.pth"

        # 3. Load Weights
        self._load_weights(self.model_ilap, path_ilap)
        self._load_weights(self.model_state, path_state)
        self._load_weights(self.gating_net, path_gate)
        
        # 4. Initialize Stepper with all models
        self.stepper = TimeStepper(
            self.model_ilap, self.model_state, self.gating_net, self.dycore, 
            self.params['mask_ori'], self.norm_cfg, self.device
        )
        
        self.model_ilap.eval()
        self.model_state.eval()
        self.gating_net.eval()

    def unpack_batch(self, batch_tuple):
        """Unpack flat list (15 elements)."""
        all_data = [x.to(self.device, non_blocking=True).float() for x in batch_tuple]
        coarse_data = all_data[0:6]  # T, S, U, V, Eta, A
        fine_data = all_data[9:15]   # T, S, U, V, Eta, A
        return fine_data, coarse_data

    def crop_and_regridding(self, tensor):
        """Replicates train_hom.py logic for Coarse data"""
        part1 = tensor[..., 200:310, 580:720]
        part2 = tensor[..., 200:310, 0:80]
        cropped = torch.cat([part1, part2], dim=-1)
        
        input_shape = cropped.shape
        H_new, W_new = 180, 360
        
        flattened = cropped.reshape(-1, 1, 110, 220)
        resized = F.interpolate(flattened, size=(H_new, W_new), mode='bilinear', align_corners=False)
        
        final_shape = input_shape[:-2] + (H_new, W_new)
        return resized.reshape(final_shape)

    def run_inference_loop(self):
        test_steps = self.train_cfg.get('integral_interval_test', 60)
        pbar = tqdm(self.test_loader, desc="Inference", disable=not self.is_master)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                # 1. Unpack & Preprocess
                fine_data, coarse_data = self.unpack_batch(batch_data)
                
                (f_T, f_S, f_u, f_v, f_eta, f_A) = fine_data
                (c_T, c_S, c_u, c_v, c_eta, c_A) = coarse_data
                
                c_T = self.crop_and_regridding(c_T)
                c_S = self.crop_and_regridding(c_S)
                c_u = self.crop_and_regridding(c_u)
                c_v = self.crop_and_regridding(c_v)
                c_eta = self.crop_and_regridding(c_eta)
                c_A = self.crop_and_regridding(c_A)

                curr_T, curr_S = f_T[:, 0], f_S[:, 0]
                curr_u, curr_v = f_u[:, 0], f_v[:, 0]
                curr_eta = f_eta[:, 0]

                # CPU Output Containers
                res_T, res_S = [curr_T.cpu()], [curr_S.cpu()]
                res_u, res_v = [curr_u.cpu()], [curr_v.cpu()]
                res_eta = [curr_eta.cpu()]

                # 2. Rollout Loop
                for idx in range(test_steps):
                    c_T_curr, c_S_curr = c_T[:, idx], c_S[:, idx]
                    c_u_curr, c_v_curr = c_u[:, idx], c_v[:, idx]
                    c_eta_curr = c_eta[:, idx]
                    c_A_curr = c_A[:, idx] 

                    c_T_next, c_S_next = c_T[:, idx+1], c_S[:, idx+1]
                    c_u_next, c_v_next = c_u[:, idx+1], c_v[:, idx+1]
                    c_eta_next = c_eta[:, idx+1]
                    c_A_next = c_A[:, idx+1] 

                    # Attention Slice (Wind) logic matching train script
                    curr_A_slice = torch.cat([f_A[:, idx], f_A[:, idx+1]], dim=1)

                    # Pass Step Index for Damping
                    curr_T, curr_S, curr_u, curr_v, curr_eta = self.stepper(
                        curr_T, curr_S, curr_u, curr_v, curr_eta, curr_A_slice,
                        c_T_curr, c_S_curr, c_u_curr, c_v_curr, c_eta_curr, c_A_curr,
                        c_T_next, c_S_next, c_u_next, c_v_next, c_eta_next, c_A_next,
                        step_idx=idx 
                    )

                    res_T.append(curr_T.cpu())
                    res_S.append(curr_S.cpu())
                    res_u.append(curr_u.cpu())
                    res_v.append(curr_v.cpu())
                    res_eta.append(curr_eta.cpu())

                # 3. Stack & Smooth
                out_T = torch.stack(res_T, dim=1)
                out_S = torch.stack(res_S, dim=1)
                out_u = torch.stack(res_u, dim=1)
                out_v = torch.stack(res_v, dim=1)
                out_eta = torch.stack(res_eta, dim=1)
                
                # Inputs (No smoothing for inputs)
                gt_T = f_T.cpu(); gt_S = f_S.cpu()
                gt_u = f_u.cpu(); gt_v = f_v.cpu()
                gt_eta = f_eta.cpu()

                batch_size = out_T.shape[0]
                world_size = self.num_gpus
                
                for i in range(batch_size):
                    global_idx = (batch_idx * world_size * batch_size) + (self.local_rank * batch_size) + i
                    
                    # --- Save Input ---
                    # Format: [Time, Channel, H, W]
                    in_save = torch.cat([
                        gt_T[i, :test_steps+1], gt_S[i, :test_steps+1], 
                        gt_u[i, :test_steps+1], gt_v[i, :test_steps+1], 
                        gt_eta[i, :test_steps+1].unsqueeze(1)
                    ], dim=1)
                    
                    # --- Process Output with Smoothing ---
                    t_np = out_T[i].numpy()
                    s_np = out_S[i].numpy()
                    u_np = out_u[i].numpy()
                    v_np = out_v[i].numpy()
                    e_np = out_eta[i].unsqueeze(1).numpy()
                    
                    # Apply Dynamic Smoothing
                    t_smooth = smooth_regional_data_dynamic(t_np, max_sigma=1.0, start_day=11)
                    s_smooth = smooth_regional_data_dynamic(s_np, max_sigma=1.0, start_day=11)
                    u_smooth = smooth_regional_data_dynamic(u_np, max_sigma=1.0, start_day=11)
                    v_smooth = smooth_regional_data_dynamic(v_np, max_sigma=1.0, start_day=11)
                    e_smooth = smooth_regional_data_dynamic(e_np, max_sigma=1.0, start_day=11)
                    
                    # Concatenate for Saving
                    out_save = np.concatenate([
                        t_smooth, s_smooth, u_smooth, v_smooth, e_smooth
                    ], axis=1)
                    
                    save_name_in = self.result_dir / f"{self.log_cfg['backbone']}/{self.log_cfg['backbone']}_input_{global_idx:06d}.npy"
                    save_name_out = self.result_dir / f"{self.log_cfg['backbone']}/{self.log_cfg['backbone']}_output_{global_idx:06d}.npy"
                    
                    save_name_in.parent.mkdir(parents=True, exist_ok=True)
                    
                    np.save(save_name_in, in_save.numpy())
                    np.save(save_name_out, out_save)
                
                if self.is_master and batch_idx % 5 == 0:
                    logging.info(f"Processed batch {batch_idx}")

        if self.parallel_method == 'DistributedDataParallel':
            dist.barrier()
            
        if self.is_master:
            logging.info(f"Inference completed. Results saved to {self.result_dir}")

if __name__ == "__main__":
    inferrer = HybridOMInferrer()
    inferrer.run_inference_loop()