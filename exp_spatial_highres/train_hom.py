import sys
import os
import logging
import random
import gc
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint
import torch.utils.data as data_utils
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from pathlib import Path

# Add project root to path (portable; avoids hard-coded personal/cluster paths)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Custom Imports
from utils.dynamical_core_regional import DynamicalCore, create_hom_parameters
from dataloader import TrainDataset, ValDataset, TestDataset

# Import ParamNet components.
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

def reduce_mean(tensor, nprocs):
    if nprocs <= 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class TimeStepper(nn.Module):
    """
    Refactored TimeStepper based on PhyNest logic (No Sea Ice).
    dX_h/dt + div(F_theta) = S_theta
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
        """Normalization logic from train_hom_old for the full state stack"""
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
        # indices mapping: (tensor_channel_idx, constant_idx)
        # Note: 49, 50, 51 usually correspond to Atmos variables in the full stack
        indices = [(49, 0), (50, 1), (51, 2), (52, 0), (53, 1), (54, 2)]
        for idx_tensor, idx_const in indices:
            s[:,:,idx_tensor] = (s[:,:,idx_tensor] - self.norm_sub[idx_const]) / self.norm_div_last[idx_const]
        return s

    def _normalize_atmos(self, atmos_tensor):
        """
        [New] Helper specifically for normalizing standalone Atmos tensors (c_A).
        Shape of atmos_tensor: (B, 3, H, W)
        Uses norm_sub[0,1,2] and norm_div_last[0,1,2] corresponding to the 3 atmos channels.
        """
        s = atmos_tensor.clone()
        # Channel 0 -> norm index 0
        s[:, 0] = (s[:, 0] - self.norm_sub[0]) / self.norm_div_last[0]
        # Channel 1 -> norm index 1
        s[:, 1] = (s[:, 1] - self.norm_sub[1]) / self.norm_div_last[1]
        # Channel 2 -> norm index 2
        s[:, 2] = (s[:, 2] - self.norm_sub[2]) / self.norm_div_last[2]
        return s

    def forward(self, 
                # Fine Grid State (t) - X^h_t
                f_T, f_S, f_u, f_v, f_eta, f_A,
                # Coarse Grid State (t) - X^l_t
                c_T_curr, c_S_curr, c_u_curr, c_v_curr, c_eta_curr, c_A_curr,
                # Coarse Grid State (t+1) - X^l_{t+1}
                c_T_next, c_S_next, c_u_next, c_v_next, c_eta_next, c_A_next):
        
        # --- 1. Prepare Placeholders ---
        B, D, H, W = f_T.shape
        
        # Sea Ice Placeholders (Required by Dycore API)
        zero_ice_2d = torch.zeros((B, H, W), device=self.device)
        zero_ice_3d = torch.zeros((B, 1, H, W), device=self.device)
        
        # Mean Fields Placeholders (Required by Dycore API)
        zero_mean_T = torch.zeros_like(f_T)
        zero_mean_S = torch.zeros_like(f_S)
        zero_mean_eta = torch.zeros((B, H, W), device=self.device)
        
        # =====================================================================
        # Step 2: Calculate Flux Terms using dycore API
        # =====================================================================
        
        # 1. Flux F(X_t^h) - Fine Current
        f_flux_T, f_flux_S = self.dycore.compute_flux(f_u, f_v, f_T, f_S)
        
        # 2. Flux F(X_t^l) - Coarse Current
        with torch.no_grad():
            c_flux_T_curr, c_flux_S_curr = self.dycore.compute_flux(c_u_curr, c_v_curr, c_T_curr, c_S_curr)
            c_flux_T_next, c_flux_S_next = self.dycore.compute_flux(c_u_next, c_v_next, c_T_next, c_S_next)

        # =====================================================================
        # Step 3: Calculate Fused Flux (F_theta) using Gated NN
        # =====================================================================
        
        # Normalize atmospheric forcing before passing to the gating network.
        norm_c_A_curr = self._normalize_atmos(c_A_curr)
        norm_c_A_next = self._normalize_atmos(c_A_next)

        fused_flux_T, fused_flux_S = self.gating_net(
            f_flux_T, c_flux_T_curr, c_flux_T_next,
            f_flux_S, c_flux_S_curr, c_flux_S_next,
            norm_c_A_curr, norm_c_A_next  # Using normalized inputs
        )

        scale_factor_state = 0.001
        intermediate_T = f_T + self.dycore.compute_div(fused_flux_T) * self.day2sec * scale_factor_state
        intermediate_S = f_S + self.dycore.compute_div(fused_flux_S) * self.day2sec * scale_factor_state
        
        # =====================================================================
        # Step 4 (Part A): U, V, Eta Forward (Physics only, no ILAP)
        # =====================================================================
        
        # 1. Run Standard Dycore Forward for Fine Grid
        # Get next_vort_phy for ILAP
        _, _, next_vort_phy, _, _, _ = self.dycore(
            f_u, f_v, f_T, f_S, f_eta,  
            zero_ice_3d, zero_ice_3d, zero_ice_2d, zero_ice_2d,
            zero_mean_T, zero_mean_S, zero_mean_eta 
        )
        
        ilap_input = torch.cat([
            next_vort_phy.unsqueeze(1) * 1e4,  # Scale vorticity
            f_u.unsqueeze(1),
            f_v.unsqueeze(1),
            f_A.unsqueeze(1)
        ], dim=2) 
        
        # 3. ILAP Prediction
        next_uv_correction = self.model_ilap(ilap_input)
        
        # 4. Apply ILAP correction
        num_depth = f_u.shape[1]
        delta_u = next_uv_correction[:, 0, 0:num_depth]
        delta_v = next_uv_correction[:, 0, num_depth:2*num_depth]
        
        scale_factor_ilap = 0.02
        
        intermediate_u = f_u + delta_u * scale_factor_ilap
        intermediate_v = f_v + delta_v * scale_factor_ilap
        intermediate_eta = f_eta 

        # =====================================================================
        # Step 4 (Part B): State Network Correction
        # =====================================================================

        # Multi-resolution Stack
        # 1. High Res Current
        high_res_stack = torch.cat([
            f_T, f_S, f_u, f_v, 
            f_eta.unsqueeze(1), f_A
        ], dim=1)  # [B, 50, H, W]
        
        # 2. Low Res Current
        low_res_curr_stack = torch.cat([
            c_T_curr, c_S_curr, c_u_curr, c_v_curr, 
            c_eta_curr.unsqueeze(1)
        ], dim=1)  # [B, 49, H, W]
        
        # 3. Low Res Next
        low_res_next_stack = torch.cat([
            c_T_next, c_S_next, c_u_next, c_v_next, 
            c_eta_next.unsqueeze(1)
        ], dim=1)  # [B, 49, H, W]
        
        # 4. Low Res Delta
        low_res_delta_stack = low_res_next_stack - low_res_curr_stack
        
        # 5. Combined Input
        combined_input = torch.cat([
            high_res_stack,          # High Res Current
            low_res_delta_stack      # Low Res Delta
        ], dim=1)  # [B, 197, H, W]
        
        # Add time dimension for 3D Conv
        combined_input = combined_input.unsqueeze(1)  # [B, 1, 197, H, W]
        
        # Normalize
        norm_combined_input = self._normalize(combined_input)
        
        # Neural Network Correction
        scale_factor_neural = 0.02 
        S_theta = self.model_state(norm_combined_input) * scale_factor_neural
        
        mask_expanded = self.mask_ori.unsqueeze(0).expand(f_T.shape[0], -1, -1, -1)
        
        final_T = (intermediate_T + S_theta[:, 0, 0:12]) * mask_expanded
        final_S = (intermediate_S + S_theta[:, 0, 12:24]) * mask_expanded
        final_u = (intermediate_u + S_theta[:, 0, 24:36]) * mask_expanded
        final_v = (intermediate_v + S_theta[:, 0, 36:48]) * mask_expanded
        final_eta = intermediate_eta + S_theta[:, 0, 48]
        
        # Boundaries
        final_T[..., 0] = final_T[..., -1]
        final_S[..., 0] = final_S[..., -1]
        final_u[..., 0] = final_u[..., -1]
        final_v[..., 0] = final_v[..., -1]
        final_eta[..., 0] = final_eta[..., -1]

        return final_T, final_S, final_u, final_v, final_eta, next_vort_phy


class HybridOMTrainer:
    def __init__(self, config_path='config.yaml'):
        self.load_config(config_path)
        self.setup_ddp()
        self.setup_logging()
        self.setup_data_and_physics()
        self.setup_models()
        self.setup_optimizer()
        
    def load_config(self, path):
        with open(path, 'r') as f:
            self.full_config = yaml.safe_load(f)
        
        self.model_name = self.full_config['selected_model']
        self.model_cfg = self.full_config['models'][self.model_name]
        self.train_cfg = self.full_config['trainings'][self.model_name]
        self.log_cfg = self.full_config['loggings'][self.model_name]
        self.norm_cfg = self.model_cfg['normalization']
        
        set_seed(self.train_cfg['seed'])
        
        self.log_dir = Path(self.log_cfg['log_dir'])
        self.ckpt_dir = Path(self.log_cfg['checkpoint_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def setup_ddp(self):
        self.parallel_method = self.train_cfg.get('parallel_method', 'DistributedDataParallel')
        if self.parallel_method == 'DistributedDataParallel':
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
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            log_format = '%(asctime)s %(levelname)s: %(message)s'
            formatter = logging.Formatter(log_format)
            log_file = f"{self.log_dir}/{self.log_cfg['backbone']}_training_log.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.handlers.clear()
            root_logger.addHandler(file_handler)
            logging.info(f"Initialized training on {self.num_gpus} GPUs.")
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
        
        self.train_loader = data_utils.DataLoader(
            TrainDataset(), 
            num_workers=1, batch_size=self.train_cfg['batch_size'], pin_memory=True,
            sampler=DistributedSampler(TrainDataset()) if self.parallel_method == 'DistributedDataParallel' else None
        )
        self.val_loader = data_utils.DataLoader(
            ValDataset(),
            num_workers=1, batch_size=self.train_cfg['batch_size'], pin_memory=True, shuffle=False,
            sampler=DistributedSampler(ValDataset(), shuffle=False) if self.parallel_method == 'DistributedDataParallel' else None
        )
        self.test_loader = data_utils.DataLoader(
            TestDataset(),
            num_workers=1, batch_size=self.train_cfg['batch_size'], pin_memory=True, shuffle=False,
            sampler=DistributedSampler(TestDataset(), shuffle=False) if self.parallel_method == 'DistributedDataParallel' else None
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

    def setup_models(self):
        # 1. Initialize Models (Random Initialization)
        model_cls = OceanDynamicsModel 
        
        # ILAP Model
        self.model_ilap = model_cls(**self.model_cfg['odm_params_ilap']).to(self.device)
        
        # State Model
        self.model_state = model_cls(**self.model_cfg['odm_params_state']).to(self.device)
        
        # Gating Network
        self.gating_net = FluxGatingUnit(**self.model_cfg['gating_params']).to(self.device)

        if self.is_master:
            logging.info("Models initialized.")

        # 2. Stepper
        self.stepper = TimeStepper(
            self.model_ilap, self.model_state, self.gating_net, self.dycore,
            self.params['mask_ori'], self.norm_cfg, self.device
        )
        
        # 3. Distributed Data Parallel
        if self.parallel_method == 'DistributedDataParallel':
            self.model_ilap = nn.parallel.DistributedDataParallel(
                self.model_ilap, device_ids=[self.local_rank], output_device=self.local_rank
            )
            self.gating_net = nn.parallel.DistributedDataParallel(
                self.gating_net, device_ids=[self.local_rank], output_device=self.local_rank
            )
            self.model_state = nn.parallel.DistributedDataParallel(
                self.model_state, device_ids=[self.local_rank], output_device=self.local_rank
            )

    def setup_optimizer(self):
        self.criterion = nn.MSELoss()
        lr = float(self.train_cfg.get('init_lr', 1e-3))
        
        # Optimize all learnable components.
        params = list(self.model_ilap.parameters()) + list(self.model_state.parameters()) + list(self.gating_net.parameters())
        
        self.optimizer = optim.Adam(params, lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_cfg['num_epochs'], eta_min=1e-8
        )

    def unpack_batch(self, batch_tuple):
        """
        Unpack flat list from dataloader (No Ice Variables).
        Total Length: 15 (6 Coarse + 3 Mean + 6 Fine)
        """
        all_data = [x.to(self.device, non_blocking=True).float() for x in batch_tuple]
        coarse_data = all_data[0:6]  # T, S, U, V, Eta, A
        fine_data = all_data[9:15]   # T, S, U, V, Eta, A
        return fine_data, coarse_data

    def crop_and_regridding(self, tensor):
        """
        New Logic: Crop global/coarse data and regrid to regional size.
        """
        part1 = tensor[..., 200:310, 580:720]
        part2 = tensor[..., 200:310, 0:80]
        cropped = torch.cat([part1, part2], dim=-1) # [..., 110, 220]
        
        input_shape = cropped.shape
        H_new, W_new = 180, 360
        
        flattened = cropped.reshape(-1, 1, 110, 220)
        
        resized = F.interpolate(
            flattened, 
            size=(H_new, W_new), 
            mode='bilinear', 
            align_corners=False
        )
        
        final_shape = input_shape[:-2] + (H_new, W_new)
        return resized.reshape(final_shape)

    def run_rollout(self, fine_inputs, coarse_inputs, integral_interval, use_checkpointing=False):
        # Unpack Fine Inputs (Initial State at t=0)
        (f_T, f_S, f_u, f_v, f_eta, f_A) = fine_inputs
        
        # Unpack Coarse Inputs (Full Time Sequence from dataloader)
        (c_T, c_S, c_u, c_v, c_eta, c_A) = coarse_inputs
        
        # Apply Crop and Regridding to ALL Coarse Inputs
        c_T = self.crop_and_regridding(c_T)
        c_S = self.crop_and_regridding(c_S)
        c_u = self.crop_and_regridding(c_u)
        c_v = self.crop_and_regridding(c_v)
        c_eta = self.crop_and_regridding(c_eta)
        c_A = self.crop_and_regridding(c_A) 

        # Initialize Current Fine State
        curr_T, curr_S = f_T[:, 0], f_S[:, 0]
        curr_u, curr_v = f_u[:, 0], f_v[:, 0]
        curr_eta = f_eta[:, 0]
        
        # Output Containers
        out_T = torch.zeros_like(f_T); out_T[:,0] = curr_T
        out_S = torch.zeros_like(f_S); out_S[:,0] = curr_S
        out_u = torch.zeros_like(f_u); out_u[:,0] = curr_u
        out_v = torch.zeros_like(f_v); out_v[:,0] = curr_v
        out_eta = torch.zeros_like(f_eta); out_eta[:,0] = curr_eta

        for idx in range(integral_interval):
            # Step 1: Get X^l from dataloader (Now cropped and regridded)
            c_T_curr, c_S_curr = c_T[:, idx], c_S[:, idx]
            c_u_curr, c_v_curr = c_u[:, idx], c_v[:, idx]
            c_eta_curr = c_eta[:, idx]
            c_A_curr = c_A[:, idx] 

            c_T_next, c_S_next = c_T[:, idx+1], c_S[:, idx+1]
            c_u_next, c_v_next = c_u[:, idx+1], c_v[:, idx+1]
            c_eta_next = c_eta[:, idx+1]
            c_A_next = c_A[:, idx+1] 

            # Current Fine Grid Attention Slice (Wind)
            curr_A_slice = torch.cat([f_A[:, idx], f_A[:, idx+1]], dim=1) 

            # Stepper Forward
            if use_checkpointing and idx > 0:
                step_res = checkpoint.checkpoint(
                    self.stepper,
                    curr_T, curr_S, curr_u, curr_v, curr_eta, curr_A_slice,
                    c_T_curr, c_S_curr, c_u_curr, c_v_curr, c_eta_curr, c_A_curr,
                    c_T_next, c_S_next, c_u_next, c_v_next, c_eta_next, c_A_next,
                    use_reentrant=False
                )
            else:
                step_res = self.stepper(
                    curr_T, curr_S, curr_u, curr_v, curr_eta, curr_A_slice,
                    c_T_curr, c_S_curr, c_u_curr, c_v_curr, c_eta_curr, c_A_curr,
                    c_T_next, c_S_next, c_u_next, c_v_next, c_eta_next, c_A_next
                )
            
            # Update States
            # Stepper returns an extra diagnostic (vorticity).
            (curr_T, curr_S, curr_u, curr_v, curr_eta, _) = step_res

            # Save Results
            out_T[:, idx+1] = curr_T
            out_S[:, idx+1] = curr_S
            out_u[:, idx+1] = curr_u
            out_v[:, idx+1] = curr_v
            out_eta[:, idx+1] = curr_eta

        return out_T, out_S, out_u, out_v, out_eta

    def compute_loss(self, pred, target):
        p_T, p_S, p_u, p_v, p_eta = pred
        # Target: Fine Grid Ground Truth (T, S, U, V, Eta)
        t_T, t_S, t_u, t_v, t_eta = target[0], target[1], target[2], target[3], target[4]
        
        loss = (
            self.criterion(p_T[:, 1:], t_T[:, 1:]) +
            self.criterion(p_S[:, 1:], t_S[:, 1:]) +
            6 * self.criterion(p_u[:, 1:], t_u[:, 1:]) +
            6 * self.criterion(p_v[:, 1:], t_v[:, 1:]) +
            6 * self.criterion(p_eta[:, 1:], t_eta[:, 1:])
        )
        print(f'l_T:{self.criterion(p_T[:, 1:], t_T[:, 1:])}, l_S:{self.criterion(p_S[:, 1:], t_S[:, 1:])}, l_u:{self.criterion(p_u[:, 1:], t_u[:, 1:])}, l_v:{self.criterion(p_v[:, 1:], t_v[:, 1:])}, l_eta:{self.criterion(p_eta[:, 1:], t_eta[:, 1:])}')
        return loss

    def cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_epoch(self, epoch):
        # Train mode for all components.
        self.model_ilap.train()
        self.gating_net.train()
        self.model_state.train()
        
        if self.parallel_method == 'DistributedDataParallel':
            self.train_loader.sampler.set_epoch(epoch)
            
        epoch_loss = 0.0
        train_steps = self.train_cfg.get('integral_interval', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}", disable=not self.is_master)
        
        for batch_tuple in pbar:
            fine_data, coarse_data = self.unpack_batch(batch_tuple)
            targets = (fine_data[0], fine_data[1], fine_data[2], fine_data[3], fine_data[4])
            
            self.optimizer.zero_grad()
            preds = self.run_rollout(fine_data, coarse_data, integral_interval=train_steps, use_checkpointing=False)
            
            loss = self.compute_loss(preds, targets)
            loss.backward()
            self.optimizer.step()
            
            reduced_loss = reduce_mean(loss, self.num_gpus).item()
            # print(reduced_loss)
            epoch_loss += reduced_loss * fine_data[0].size(0)
            
            self.cleanup_memory()

        return epoch_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model_ilap.eval()
        self.gating_net.eval()
        self.model_state.eval()
        
        val_loss = 0.0
        val_steps = self.train_cfg.get('integral_interval', 1) 
        
        with torch.no_grad():
            for batch_tuple in tqdm(self.val_loader, desc="Validating", disable=not self.is_master):
                fine_data, coarse_data = self.unpack_batch(batch_tuple)
                targets = (fine_data[0], fine_data[1], fine_data[2], fine_data[3], fine_data[4])
                
                preds = self.run_rollout(fine_data, coarse_data, integral_interval=val_steps, use_checkpointing=False)
                loss = self.compute_loss(preds, targets)
                
                reduced_loss = reduce_mean(loss, self.num_gpus).item()
                val_loss += reduced_loss * fine_data[0].size(0)
                
        return val_loss / len(self.val_loader.dataset)

    def test(self):
        self.model_ilap.eval()
        self.gating_net.eval()
        self.model_state.eval()
        
        test_steps = self.train_cfg.get('integral_interval_test', 60)
        local_step_loss_sum = torch.zeros(test_steps, device=self.device)
        local_sample_count = torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            for batch_tuple in tqdm(self.test_loader, desc="Testing", disable=not self.is_master):
                fine_data, coarse_data = self.unpack_batch(batch_tuple)
                current_batch_size = fine_data[0].size(0)
                
                preds = self.run_rollout(fine_data, coarse_data, integral_interval=test_steps, use_checkpointing=False)
                
                p_T, p_S, p_u, p_v, p_eta = preds
                t_T, t_S, t_u, t_v, t_eta = fine_data[0], fine_data[1], fine_data[2], fine_data[3], fine_data[4]
                
                l_T = (p_T[:, 1:] - t_T[:, 1:]).pow(2).mean(dim=(0, 2, 3, 4)) 
                l_S = (p_S[:, 1:] - t_S[:, 1:]).pow(2).mean(dim=(0, 2, 3, 4))
                l_u = (p_u[:, 1:] - t_u[:, 1:]).pow(2).mean(dim=(0, 2, 3, 4))
                l_v = (p_v[:, 1:] - t_v[:, 1:]).pow(2).mean(dim=(0, 2, 3, 4))
                l_eta = (p_eta[:, 1:] - t_eta[:, 1:]).pow(2).mean(dim=(0, 2, 3))
                
                step_loss_vec = l_T + l_S + 6 * l_u + 6 * l_v + 6 * l_eta
                local_step_loss_sum += step_loss_vec * current_batch_size
                local_sample_count += current_batch_size

        if self.parallel_method == 'DistributedDataParallel':
            dist.all_reduce(local_step_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sample_count, op=dist.ReduceOp.SUM)

        avg_step_losses = local_step_loss_sum / local_sample_count
        avg_total_loss = avg_step_losses.mean().item()

        if self.is_master:
            logging.info(f"Test Completed. Mean Loss: {avg_total_loss:.6f}")

        return avg_total_loss

    def save_checkpoint(self, is_best=False):
        if not self.is_master: return
        backbone = self.log_cfg['backbone']
        # Save ILAP, gating, and state network weights.
        name_ilap = f"{backbone}_ilap_best.pth"
        name_gating = f"{backbone}_gating_best.pth"
        name_state = f"{backbone}_state_best.pth"
        
        # Unwrap DDP models for saving
        ilap_dict = self.model_ilap.module.state_dict() if hasattr(self.model_ilap, 'module') else self.model_ilap.state_dict()
        gating_state_dict = self.gating_net.module.state_dict() if hasattr(self.gating_net, 'module') else self.gating_net.state_dict()
        state_model_dict = self.model_state.module.state_dict() if hasattr(self.model_state, 'module') else self.model_state.state_dict()
        
        torch.save(ilap_dict, self.ckpt_dir / name_ilap)
        torch.save(gating_state_dict, self.ckpt_dir / name_gating)
        torch.save(state_model_dict, self.ckpt_dir / name_state)

    def run(self):
        best_loss = float('inf')
        for epoch in range(self.train_cfg['num_epochs']):
            if self.is_master: logging.info(f"Starting Epoch {epoch + 1}")
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            if self.is_master:
                curr_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch+1} | LR: {curr_lr:.8f} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(is_best=True)
        if self.parallel_method == 'DistributedDataParallel':
            dist.destroy_process_group()

if __name__ == "__main__":
    trainer = HybridOMTrainer()
    trainer.run()
