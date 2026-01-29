import sys
import os
import logging
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
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
from utils.dynamical_core import DynamicalCore, create_hom_parameters
from dataloader import TestDataset  # Only need TestDataset for inference
from networks.baseline.baseline_ST.simvp import SimVP
from networks.baseline.baseline_ST.turb_l1 import TurbL1
from networks.baseline.baseline_ST.ParamNet import OceanDynamicsModel

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
# Time-dependent Gaussian smoothing (for visualization / long-rollout stabilization).
# ==============================================================================

def smooth_global_data_dynamic(data, max_sigma=0.8, start_day=10, full_day=40):
    """
    Apply time-dependent Gaussian smoothing to a 4D (Time, Channel, Lat, Lon) 
    global array.
    
    Logic:
    - t < start_day: sigma = 0
    - start_day <= t < full_day: sigma ramps linearly from 0 to max_sigma
    - t >= full_day: sigma = max_sigma
    """
    if data.ndim != 4:
        raise ValueError(f"Input data must be 4D (T, C, H, W). Got shape {data.shape}")
    
    T, C, H, W = data.shape
    output = np.zeros_like(data)
    
    # Calculate the duration of the ramping phase
    ramp_duration = full_day - start_day
    
    for t in range(T):
        # 1. Calculate dynamic sigma
        if t < start_day:
            current_sigma = 0.0
        elif t >= full_day:
            # After full_day, keep sigma constant at max_sigma
            current_sigma = max_sigma
        else:
            current_sigma = 0.5 + (max_sigma - 0.5) * (t - start_day) / ramp_duration 

        # 2. Skip if sigma is negligible
        if current_sigma < 1e-6:
            output[t] = data[t]
            continue
        # print(current_sigma)
        # 3. Padding logic
        # Calculate padding width based on sigma (3-sigma rule)
        pad_width = int(np.ceil(current_sigma * 3))
        
        # Pad only the last dimension (Longitude) with wrap to handle periodicity
        # data[t] shape is (C, H, W) -> pad axes: ((0,0), (0,0), (pad, pad))
        pad_axes = ((0, 0), (0, 0), (pad_width, pad_width))
        
        data_t = data[t]
        data_padded = np.pad(data_t, pad_axes, mode='wrap')
        
        # 4. Define sigma tuple (0 for channels, sigma for H/W)
        # Apply smoothing only on spatial dimensions
        sigma_tuple = (0, current_sigma, current_sigma)
        
        # 5. Apply Filter (nearest mode for Latitude boundary to avoid artifacts)
        smoothed_padded = gaussian_filter(data_padded, sigma=sigma_tuple, mode='nearest')
        
        # 6. Crop back to original size (remove padding)
        output[t] = smoothed_padded[:, :, pad_width:-pad_width]

    return output
# ==============================================================================

class TimeStepper(nn.Module):
    """
    Encapsulates a single physical-neural time step.
    Copied from train_hom.py to ensure exact consistency in normalization and logic.
    """
    def __init__(self, model_ilap, model_state, dycore, mask_ori, norm_cfg, device):
        super().__init__()
        self.model_ilap = model_ilap
        self.model_state = model_state
        self.dycore = dycore
        self.register_buffer('mask_ori', mask_ori)
        self.device = device

        # --- Load Normalization Constants as Buffers ---
        self.register_buffer('norm_div', torch.tensor(norm_cfg['div'], dtype=torch.float32))
        self.register_buffer('norm_sub', torch.tensor(norm_cfg['sub'], dtype=torch.float32))
        self.register_buffer('norm_div_last', torch.tensor(norm_cfg['div_last'], dtype=torch.float32))

    def _normalize(self, state):
        s = state.clone()
        
        # Division normalization for first 49 channels
        # self.norm_div shape is [5]. 
        s[:,:,0:23]   /= self.norm_div[0]
        s[:,:,23:46]  /= self.norm_div[1]
        s[:,:,46:69]  /= self.norm_div[2]
        s[:,:,69:92]  /= self.norm_div[3]
        s[:,:,92]     /= self.norm_div[4]
        
        # Shift-and-scale normalization for auxiliary channels.
        # Indices: 49(sic), 50(sit), 51(usi), 52(vsi) -> Always Normalize
        indices_common = [(93, 0), (94, 1), (95, 2), (96, 0), (97, 1), (98, 2)]
        for idx_tensor, idx_const in indices_common:
            s[:,:,idx_tensor] = (s[:,:,idx_tensor] - self.norm_sub[idx_const]) / self.norm_div_last[idx_const]
        
        # Indices: 53(A0), 54(A1) -> Normalize ONLY if using old logic (NeuralOM)
        # If forcing_type is 'WenHai', data is already normalized.
        # if self.forcing_type != 'WenHai':
        #     indices_forcing = [(53, 1), (54, 2)]
        #     for idx_tensor, idx_const in indices_forcing:
        #         s[:,:,idx_tensor] = (s[:,:,idx_tensor] - self.norm_sub[idx_const]) / self.norm_div_last[idx_const]
            
        return s
    
    def _get_damping_factor(self, step_idx, start_val, end_val, start_step=10, end_step=60):
        """Helper to calculate linear ramp damping factor."""
        if step_idx >= end_step:
            return end_val
        else:
            progress = (step_idx - start_step) / (end_step - start_step)
            return start_val - progress * (start_val - end_val)

    def forward(self, current_T, current_S, current_u, current_v, current_eta, 
                current_sic, current_sit, current_usi, current_vsi, current_A_slice,
                current_T_mean, current_S_mean, current_eta_mean, step_idx=0): # Added step_idx
        # 1. Physics Step
        next_T, next_S, next_vort, current_vort, next_sic, next_sit = self.dycore(
            current_u, current_v, current_T, current_S, current_eta[:,0], 
            current_sic, current_sit, current_usi, current_vsi, 
            current_T_mean, current_S_mean, current_eta_mean
        )

        # 2. ILAP Prediction
        next_vort_stacked = torch.cat([
            next_vort.unsqueeze(1) * 1e4, 
            current_u.unsqueeze(1), 
            current_v.unsqueeze(1),
            current_A_slice.unsqueeze(1)
        ], dim=2)
        next_uv = self.model_ilap(next_vort_stacked)
        
        scale_factor_uv = 0.05
        next_u = current_u + next_uv[:, 0, 0:23] * scale_factor_uv
        next_v = current_v + next_uv[:, 0, 23:46] * scale_factor_uv

        # 3. State Model
        # print(current_T.shape, current_S.shape, current_u.shape, current_v.shape, 
        #     current_eta.unsqueeze(1).shape, current_A_slice.shape)
        current_state = torch.cat([
            current_T, current_S, current_u, current_v, 
            current_eta.unsqueeze(1), current_A_slice
        ], dim=1).unsqueeze(1)
        
        norm_state = self._normalize(current_state)
        scale_factor_state = 0.1
        neural_correction = self.model_state(norm_state) * scale_factor_state
        
        mask_expanded = self.mask_ori.unsqueeze(0).expand(next_T.shape[0], -1, -1, -1)
        
        # Apply Correction
        final_T = (next_T + neural_correction[:, 0, 0:23]) * mask_expanded
        final_S = (next_S + neural_correction[:, 0, 23:46]) * mask_expanded
        final_u = (next_u + neural_correction[:, 0, 46:69]) * mask_expanded
        final_v = (next_v + neural_correction[:, 0, 69:92]) * mask_expanded
        final_eta = current_eta + neural_correction[:, 0, 92]

        # ==============================================================================
        # DYNAMIC STABILIZATION: Variable-Specific Damping Schedules
        # ==============================================================================
        if step_idx >= 11:
            # 1. Calculate Tendencies (Net change)
            delta_T = final_T - current_T
            delta_S = final_S - current_S
            delta_u = final_u - current_u
            delta_v = final_v - current_v
            # delta_eta = final_eta - current_eta
            
            # 2. Calculate Damping Factors
            # Schedule 1: For T, u, v -> 0.90 down to 0.50
            damp_temp = self._get_damping_factor(step_idx, start_val=1.00, end_val=0.50)
            
            # Schedule 2: For S -> 0.90 down to 0.20 (Stronger suppression for Salinity)
            damp_salt = self._get_damping_factor(step_idx, start_val=1.00, end_val=0.20)
            
            # Schedule 3: For Vel, Since the results is relatively conservative, we didn't apply any damping
            damp_vel = self._get_damping_factor(step_idx, start_val=1.00, end_val=0.9)
            
            # 3. Hard Clamping Threshold (Physical constraint on step-wise change)
            clamp_ts = 0.5
            clamp_vel = 0.2
            
            # 4. Apply Stabilization (Clamp then Damp)
            # Temperature
            delta_T = torch.clamp(delta_T, -clamp_ts, clamp_ts) * damp_temp
            # Salinity (Stronger damping)
            delta_S = torch.clamp(delta_S, -clamp_ts, clamp_ts) * damp_salt
            # Velocity U
            delta_u = torch.clamp(delta_u, -clamp_vel, clamp_vel) * damp_vel 
            # Velocity V
            delta_v = torch.clamp(delta_v, -clamp_vel, clamp_vel) * damp_vel 
            # delta_eta = torch.clamp(delta_eta, -clamp_vel, clamp_vel) * damp_vel 
            
            # 5. Reconstruct Stabilized State & Re-apply Mask
            final_T = (current_T + delta_T) * mask_expanded
            final_S = (current_S + delta_S) * mask_expanded
            final_u = (current_u + delta_u) * mask_expanded
            final_v = (current_v + delta_v) * mask_expanded
            # final_eta = (current_eta + delta_eta) * mask_expanded
        # ==============================================================================

        # Boundaries
        final_T[..., 0] = final_T[..., -1]
        final_S[..., 0] = final_S[..., -1]
        final_u[..., 0] = final_u[..., -1]
        final_v[..., 0] = final_v[..., -1]
        final_eta[..., 0] = final_eta[..., -1]
        current_vort[..., 0] = current_vort[..., -1]

        return (final_T, final_S, final_u, final_v, final_eta, 
                next_sic, next_sit, current_usi, current_vsi, current_vort)


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
        
        # Load Normalization Config
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
            # Add console handler
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
        
        # NOTE: For inference, we use TestDataset
        dataset = TestDataset()
        self.sampler = DistributedSampler(dataset, shuffle=False) if self.parallel_method == 'DistributedDataParallel' else None
        
        self.test_loader = data_utils.DataLoader(
            dataset,
            num_workers=1, 
            batch_size=self.train_cfg['batch_size'], 
            pin_memory=True, 
            shuffle=False,
            sampler=self.sampler
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
        self.model_ilap = model_cls(**self.model_cfg['odm_params_ilap']).to(self.device)
        self.model_state = model_cls(**self.model_cfg['odm_params_state']).to(self.device)

        ckpt_dir = Path(self.log_cfg['checkpoint_dir'])
        backbone = self.log_cfg['backbone']
        
        # Try Loading best model from fine-tuning first
        steps = self.train_cfg.get('integral_interval', 1)
        suffix = f"_{steps}_step_supervised"
        path_state = ckpt_dir / f"{backbone}{suffix}_best_model.pth"
        path_ilap = ckpt_dir / f"ilap{suffix}_best_model.pth"

        if not path_state.exists():
             if self.is_master: logging.info("Fine-tuned weights not found, using default 5-step weights.")
             path_state = ckpt_dir / f"{backbone}_best_model.pth"
             path_ilap = ckpt_dir / f"{backbone}ilap_best_model.pth"
             
        self._load_weights(self.model_ilap, path_ilap)
        self._load_weights(self.model_state, path_state)

        self.stepper = TimeStepper(
            self.model_ilap, self.model_state, self.dycore, 
            self.params['mask_ori'], self.norm_cfg, self.device
        )
        self.model_ilap.eval()
        self.model_state.eval()

    def run_inference_loop(self):
        test_steps = self.train_cfg.get('integral_interval_test', 60)
        
        pbar = tqdm(self.test_loader, desc="Inference", disable=not self.is_master)
        
        batch_size = self.test_loader.batch_size
        world_size = self.num_gpus
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                batch_data = [x.to(self.device, non_blocking=True).float() for x in batch_data]
                
                (in_T, in_S, in_u, in_v, in_eta, in_sic, in_sit, in_usi, in_vsi, in_A, in_T_mean, in_S_mean, in_eta_mean) = batch_data
                
                curr_T, curr_S = in_T[:, 0], in_S[:, 0]
                curr_u, curr_v = in_u[:, 0], in_v[:, 0]
                curr_eta = in_eta[:, 0]
                curr_sic, curr_sit = in_sic[:, 0], in_sit[:, 0]
                curr_usi, curr_vsi = in_usi[:, 0], in_vsi[:, 0]

                # Accumulate results on CPU to save GPU memory
                res_T = [curr_T.cpu()]
                res_S = [curr_S.cpu()]
                res_u = [curr_u.cpu()]
                res_v = [curr_v.cpu()]
                res_eta = [curr_eta.cpu()]

                for idx in range(test_steps):
                    curr_A_slice = torch.cat([in_A[:, idx], in_A[:, idx+1]], dim=1)
                    curr_T_mean = 0
                    curr_S_mean = 0
                    curr_eta_mean = 0

                    step_res = self.stepper(
                        curr_T, curr_S, curr_u, curr_v, curr_eta,
                        curr_sic, curr_sit, curr_usi, curr_vsi, 
                        curr_A_slice, curr_T_mean, curr_S_mean, curr_eta_mean, 
                        step_idx=idx
                    )
                    
                    (curr_T, curr_S, curr_u, curr_v, curr_eta, 
                     curr_sic, curr_sit, curr_usi, curr_vsi, _) = step_res

                    res_T.append(curr_T.cpu())
                    res_S.append(curr_S.cpu())
                    res_u.append(curr_u.cpu())
                    res_v.append(curr_v.cpu())
                    res_eta.append(curr_eta.cpu())

                # Stack time dimension -> (Batch, Time, Channel, Lat, Lon)
                out_T = torch.stack(res_T, dim=1)
                out_S = torch.stack(res_S, dim=1)
                out_u = torch.stack(res_u, dim=1)
                out_v = torch.stack(res_v, dim=1)
                out_eta = torch.stack(res_eta, dim=1)
                
                inputs_cpu = [x.cpu() for x in batch_data]
                current_batch_count = out_T.shape[0]
                
                for i in range(current_batch_count):
                    global_idx = (batch_idx * world_size * batch_size) + (self.local_rank * batch_size) + i
                    
                    # Prepare Input (No smoothing needed usually, just raw input)
                    t_in = inputs_cpu[0][i] 
                    s_in = inputs_cpu[1][i]
                    u_in = inputs_cpu[2][i]
                    v_in = inputs_cpu[3][i]
                    e_in = inputs_cpu[4][i].unsqueeze(1)
                    input_concat = torch.cat([t_in, s_in, u_in, v_in, e_in], dim=1)
                    
                    # ----------------------------------------------------------
                    # Convert outputs to NumPy and apply time-dependent smoothing.
                    # Shapes: (Time, Channel, H, W)
                    # ----------------------------------------------------------
                    t_out_np = out_T[i].numpy()
                    s_out_np = out_S[i].numpy()
                    u_out_np = out_u[i].numpy()
                    v_out_np = out_v[i].numpy()
                    # eta: (Time, H, W) -> (Time, 1, H, W)
                    e_out_np = out_eta[i].unsqueeze(1).numpy()
                    
                    # Apply smoothing (sigma ramps up after start_day).
                    t_out_smooth = smooth_global_data_dynamic(t_out_np, max_sigma=0.5, start_day=11)
                    s_out_smooth = smooth_global_data_dynamic(s_out_np, max_sigma=0.7, start_day=11)
                    u_out_smooth = smooth_global_data_dynamic(u_out_np, max_sigma=0.7, start_day=11)
                    v_out_smooth = smooth_global_data_dynamic(v_out_np, max_sigma=0.7, start_day=11)
                    e_out_smooth = smooth_global_data_dynamic(e_out_np, max_sigma=0.7, start_day=11)
                    
                    # Concatenate along channel dimension (axis=1).
                    output_concat = np.concatenate([
                        t_out_smooth, 
                        s_out_smooth, 
                        u_out_smooth, 
                        v_out_smooth, 
                        e_out_smooth
                    ], axis=1)
                    # ----------------------------------------------------------
                    
                    save_name_in = self.result_dir / f"{self.log_cfg['backbone']}/{self.log_cfg['backbone']}_input_{global_idx:06d}.npy"
                    save_name_out = self.result_dir / f"{self.log_cfg['backbone']}/{self.log_cfg['backbone']}_output_{global_idx:06d}.npy"
                    
                    save_name_in.parent.mkdir(parents=True, exist_ok=True)
                    
                    np.save(save_name_in, input_concat.numpy())  # input
                    np.save(save_name_out, output_concat)        # output
                
                if self.is_master and batch_idx % 10 == 0:
                    logging.info(f"Processed batch {batch_idx}, saved samples up to index {global_idx}")

        if self.parallel_method == 'DistributedDataParallel':
            dist.barrier()
            
        if self.is_master:
            logging.info(f"Inference completed. Results saved to {self.result_dir}")

if __name__ == "__main__":
    inferrer = HybridOMInferrer()
    inferrer.run_inference_loop()