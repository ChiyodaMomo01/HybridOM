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
from utils.dynamical_core_025 import DynamicalCore, create_hom_parameters
from dataloader import TrainDataset, ValDataset, TestDataset
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

def reduce_mean(tensor, nprocs):
    """Reduces a tensor across all GPUs (average)."""
    if nprocs <= 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class TimeStepper(nn.Module):
    """Encapsulates a single physical-neural time step."""
    def __init__(self, model_ilap, model_state, dycore, mask_ori, norm_cfg, device, forcing_type='NeuralOM'):
        super().__init__()
        self.model_ilap = model_ilap
        self.model_state = model_state
        self.dycore = dycore
        self.register_buffer('mask_ori', mask_ori)
        self.device = device
        self.forcing_type = forcing_type 

        # --- Load Normalization Constants as Buffers ---
        self.register_buffer('norm_div', torch.tensor(norm_cfg['div'], dtype=torch.float32))
        self.register_buffer('norm_sub', torch.tensor(norm_cfg['sub'], dtype=torch.float32))
        self.register_buffer('norm_div_last', torch.tensor(norm_cfg['div_last'], dtype=torch.float32))

    def _normalize(self, state):
        s = state.clone()
        
        # Division normalization for first 49 channels
        # self.norm_div shape is [5]. 
        s[:,:,0:12]   /= self.norm_div[0]
        s[:,:,12:24]  /= self.norm_div[1]
        s[:,:,24:36]  /= self.norm_div[2]
        s[:,:,36:48]  /= self.norm_div[3]
        s[:,:,48]     /= self.norm_div[4]
        
        # Shift and Scale logic based on forcing_type
        indices_common = [(49, 0), (50, 1), (51, 2), (52, 0), (53, 1), (54, 2)]
        for idx_tensor, idx_const in indices_common:
            s[:,:,idx_tensor] = (s[:,:,idx_tensor] - self.norm_sub[idx_const]) / self.norm_div_last[idx_const]
            
        return s

    def forward(self, current_T, current_S, current_u, current_v, current_eta, 
                current_sic, current_sit, current_usi, current_vsi, current_A_slice,
                current_T_mean, current_S_mean, current_eta_mean):
        # 1. Physics Step
        next_T, next_S, next_vort, current_vort, next_sic, next_sit = self.dycore(
            current_u, current_v, current_T, current_S, current_eta[:,0], 
            current_sic, current_sit, current_usi, current_vsi, 
            current_T_mean, current_S_mean, current_eta_mean
        )

        # 2. ILAP Prediction (velocity correction)
        # Construct input: [Vorticity, u, v, A]
        # If next_vort is 3D (B, H, W), add a channel dimension.
        if next_vort.dim() == 3:
            vort_input = next_vort.unsqueeze(1)
        else:
            vort_input = next_vort
            
        # Concatenate channels: vort + u + v + A
        ilap_input = torch.cat([
            vort_input * 1e4, 
            current_u, 
            current_v,
            current_A_slice
        ], dim=1) 
        
        # Add a dummy time dimension for ParamNet: [B, 1, C, H, W]
        ilap_input = ilap_input.unsqueeze(1)
        
        # Predict flow correction.
        next_uv_correction = self.model_ilap(ilap_input)
        
        scale_factor_uv = 0.02
        
        # Apply correction (assumes first half is u and second half is v).
        next_u = current_u + next_uv_correction[:, 0, 0:12] * scale_factor_uv
        next_v = current_v + next_uv_correction[:, 0, 12:24] * scale_factor_uv

        # 3. State Model (Residual Correction for Mass Field)
        current_state = torch.cat([
            current_T, current_S, current_u, current_v, 
            current_eta.unsqueeze(1), current_A_slice
        ], dim=1).unsqueeze(1)
        
        norm_state = self._normalize(current_state)
        
        scale_factor_state = 0.02
        neural_correction = self.model_state(norm_state) * scale_factor_state
        
        mask_expanded = self.mask_ori.unsqueeze(0).expand(next_T.shape[0], -1, -1, -1)
        
        # Apply corrections
        final_T = (next_T + neural_correction[:, 0, 0:12]) * mask_expanded
        final_S = (next_S + neural_correction[:, 0, 12:24]) * mask_expanded
        final_u = (next_u + neural_correction[:, 0, 24:36]) * mask_expanded
        final_v = (next_v + neural_correction[:, 0, 36:48]) * mask_expanded
        final_eta = current_eta + neural_correction[:, 0, 48]

        # Boundaries
        final_T[..., 0] = final_T[..., -1]
        final_S[..., 0] = final_S[..., -1]
        final_u[..., 0] = final_u[..., -1]
        final_v[..., 0] = final_v[..., -1]
        final_eta[..., 0] = final_eta[..., -1]
        current_vort[..., 0] = current_vort[..., -1]

        return (final_T, final_S, final_u, final_v, final_eta, 
                next_sic, next_sit, current_usi, current_vsi, current_vort)


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
        
        # Load Normalization Config
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
            try:
                model.load_state_dict(new_state_dict)
            except RuntimeError as e:
                if self.is_master:
                    logging.warning(f"Strict loading failed, trying non-strict. Error: {e}")
                model.load_state_dict(new_state_dict, strict=False)
        else:
            if self.is_master:
                logging.warning(f"Checkpoint path {path} does not exist!")

    def setup_models(self):
        model_cls = OceanDynamicsModel 
        
        self.model_ilap = model_cls(**self.model_cfg['odm_params_ilap']).to(self.device)
        self.model_state = model_cls(**self.model_cfg['odm_params_state']).to(self.device)

        if self.train_cfg.get('fine_tune', False):
            self._load_weights(self.model_state, self.train_cfg.get('pre_trained_path_state', ''))
            self._load_weights(self.model_ilap, self.train_cfg.get('pre_trained_path_ilap', ''))
            logging.info(f"Pretrained checkpoints loaded.")

        if self.parallel_method == 'DistributedDataParallel':
            self.model_ilap = nn.parallel.DistributedDataParallel(
                self.model_ilap, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False
            )
            self.model_state = nn.parallel.DistributedDataParallel(
                self.model_state, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False
            )
        
        forcing_type = self.train_cfg.get('forcing_type', 'NeuralOM')
        if self.is_master:
            logging.info(f"Setting up TimeStepper with forcing_type: {forcing_type}")

        self.stepper = TimeStepper(
            self.model_ilap, self.model_state, self.dycore, 
            self.params['mask_ori'], self.norm_cfg, self.device,
            forcing_type=forcing_type 
        )

    def setup_optimizer(self):
        self.criterion = nn.MSELoss()
        
        if self.train_cfg.get('fine_tune', False):
            lr = float(self.train_cfg.get('fine_tune_lr', 1e-6))
        else:
            lr = float(self.train_cfg.get('init_lr', 1e-3))
            
        # Optimize ILAP + state model parameters.
        params = list(self.model_state.parameters()) + list(self.model_ilap.parameters())
        
        self.optimizer = optim.Adam(params, lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_cfg['num_epochs'], eta_min=1e-8
        )

    def run_rollout(self, inputs, integral_interval, use_checkpointing=False):
        (in_T, in_S, in_u, in_v, in_eta, in_sic, in_sit, in_usi, in_vsi, in_A, in_T_mean, in_S_mean, in_eta_mean) = inputs
        
        curr_T, curr_S = in_T[:, 0], in_S[:, 0]
        curr_u, curr_v = in_u[:, 0], in_v[:, 0]
        curr_eta = in_eta[:, 0]
        curr_sic, curr_sit = in_sic[:, 0], in_sit[:, 0]
        curr_usi, curr_vsi = in_usi[:, 0], in_vsi[:, 0]

        out_T = torch.zeros_like(in_T); out_T[:,0] = curr_T
        out_S = torch.zeros_like(in_S); out_S[:,0] = curr_S
        out_u = torch.zeros_like(in_u); out_u[:,0] = curr_u
        out_v = torch.zeros_like(in_v); out_v[:,0] = curr_v
        out_eta = torch.zeros_like(in_eta); out_eta[:,0] = curr_eta

        for idx in range(integral_interval):
            curr_A_slice = torch.cat([in_A[:, idx], in_A[:, idx+1]], dim=1)
            curr_T_mean = 0
            curr_S_mean = 0
            curr_eta_mean = 0

            if use_checkpointing and idx > 0:
                step_res = checkpoint.checkpoint(
                    self.stepper,
                    curr_T, curr_S, curr_u, curr_v, curr_eta,
                    curr_sic, curr_sit, curr_usi, curr_vsi, 
                    curr_A_slice, curr_T_mean, curr_S_mean, curr_eta_mean,
                    use_reentrant=False
                )
            else:
                step_res = self.stepper(
                    curr_T, curr_S, curr_u, curr_v, curr_eta,
                    curr_sic, curr_sit, curr_usi, curr_vsi, 
                    curr_A_slice, curr_T_mean, curr_S_mean, curr_eta_mean
                )
            
            (curr_T, curr_S, curr_u, curr_v, curr_eta, 
             curr_sic, curr_sit, curr_usi, curr_vsi, _) = step_res

            out_T[:, idx+1] = curr_T
            out_S[:, idx+1] = curr_S
            out_u[:, idx+1] = curr_u
            out_v[:, idx+1] = curr_v
            out_eta[:, idx+1] = curr_eta

        return out_T, out_S, out_u, out_v, out_eta

    def compute_loss(self, pred, target):
        p_T, p_S, p_u, p_v, p_eta = pred
        t_T, t_S, t_u, t_v, t_eta = target
        
        loss = (
            self.criterion(p_T[:, 1:], t_T[:, 1:]) +
            self.criterion(p_S[:, 1:], t_S[:, 1:]) +
            6 * self.criterion(p_u[:, 1:], t_u[:, 1:]) +
            6 * self.criterion(p_v[:, 1:], t_v[:, 1:]) +
            6 * self.criterion(p_eta[:, 1:], t_eta[:, 1:])
        )
        # print(f'l_T:{self.criterion(p_T[:, 1:], t_T[:, 1:])}, l_S:{self.criterion(p_S[:, 1:], t_S[:, 1:])}, l_u:{self.criterion(p_u[:, 1:], t_u[:, 1:])}, l_v:{self.criterion(p_v[:, 1:], t_v[:, 1:])}, l_eta:{self.criterion(p_eta[:, 1:], t_eta[:, 1:])}')
        return loss

    def cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_epoch(self, epoch):
        self.model_state.train()
        self.model_ilap.train()
        
        if self.parallel_method == 'DistributedDataParallel':
            self.train_loader.sampler.set_epoch(epoch)
            
        epoch_loss = 0.0
        train_steps = self.train_cfg.get('integral_interval', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}", disable=not self.is_master)
        self.train_loader.dataset.reshuffle_train_samples()
        
        for batch_idx, batch_data in enumerate(pbar):
            batch_data = [x.to(self.device, non_blocking=True).float() for x in batch_data]
            targets = (batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4])
            
            self.optimizer.zero_grad()
            preds = self.run_rollout(batch_data, integral_interval=train_steps, use_checkpointing=True)
            
            loss = self.compute_loss(preds, targets)
            loss.backward()
            self.optimizer.step()
            
            reduced_loss = reduce_mean(loss, self.num_gpus).item()
            print(reduced_loss)
            
            if not np.isnan(reduced_loss):
                epoch_loss += reduced_loss * batch_data[0].size(0)
            
            self.cleanup_memory()

        return epoch_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model_state.eval()
        self.model_ilap.eval()
        val_loss = 0.0
        val_steps = self.train_cfg.get('integral_interval', 1) 
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validating", disable=not self.is_master):
                batch_data = [x.to(self.device, non_blocking=True).float() for x in batch_data]
                targets = (batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4])
                
                preds = self.run_rollout(batch_data, integral_interval=val_steps, use_checkpointing=False)
                loss = self.compute_loss(preds, targets)
                
                reduced_loss = reduce_mean(loss, self.num_gpus).item()
                val_loss += reduced_loss * batch_data[0].size(0)
                
        return val_loss / len(self.val_loader.dataset)

    def test(self):
        self.model_state.eval()
        self.model_ilap.eval()
        
        test_steps = self.train_cfg.get('integral_interval_test', 60)
        
        local_step_loss_sum = torch.zeros(test_steps, device=self.device)
        local_sample_count = torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            for batch_data in tqdm(self.test_loader, desc="Testing", disable=not self.is_master):
                batch_data = [x.to(self.device, non_blocking=True).float() for x in batch_data]
                current_batch_size = batch_data[0].size(0)
                
                p_T, p_S, p_u, p_v, p_eta = self.run_rollout(batch_data, integral_interval=test_steps, use_checkpointing=False)
                
                t_T, t_S, t_u, t_v, t_eta = batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4]
                
                p_T, p_S = p_T[:, 1:], p_S[:, 1:]
                p_u, p_v = p_u[:, 1:], p_v[:, 1:]
                p_eta = p_eta[:, 1:]
                
                t_T, t_S = t_T[:, 1:], t_S[:, 1:]
                t_u, t_v = t_u[:, 1:], t_v[:, 1:]
                t_eta = t_eta[:, 1:]
                
                l_T = (p_T - t_T).pow(2).mean(dim=(0, 2, 3, 4)) 
                l_S = (p_S - t_S).pow(2).mean(dim=(0, 2, 3, 4))
                l_u = (p_u - t_u).pow(2).mean(dim=(0, 2, 3, 4))
                l_v = (p_v - t_v).pow(2).mean(dim=(0, 2, 3, 4))
                l_eta = (p_eta - t_eta).pow(2).mean(dim=(0, 2, 3))
                
                step_loss_vec = l_T + l_S + 6 * l_u + 6 * l_v + 6 * l_eta
                
                local_step_loss_sum += step_loss_vec * current_batch_size
                local_sample_count += current_batch_size

                self.cleanup_memory()

        if self.parallel_method == 'DistributedDataParallel':
            dist.barrier()
            dist.all_reduce(local_step_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sample_count, op=dist.ReduceOp.SUM)

        avg_step_losses = local_step_loss_sum / local_sample_count
        avg_total_loss = avg_step_losses.mean().item()

        if self.is_master:
            loss_list = avg_step_losses.cpu().tolist()
            log_str = "\n" + "="*50 + "\n"
            log_str += f"Test Completed. Mean Loss: {avg_total_loss:.6f}\n"
            log_str += "Step-wise Loss (Check for explosion):\n"
            for i, loss_val in enumerate(loss_list):
                log_str += f"Step {i+1:02d}: {loss_val:.6f}  "
                if (i + 1) % 5 == 0: log_str += "\n"
            log_str += "="*50 + "\n"
            logging.info(log_str)
            print(log_str) 

        return avg_total_loss

    def save_checkpoint(self, is_best=False):
        if not self.is_master:
            return
            
        backbone = self.log_cfg['backbone']
        suffix = ""
        if self.train_cfg.get('fine_tune', False):
            steps = self.train_cfg.get('integral_interval', 1)
            suffix = f"_{steps}_step_supervised"
            
        name_state = f"{backbone}{suffix}_best_model.pth"
        name_ilap = f"ilap{suffix}_best_model.pth"
        
        state_dict = self.model_state.module.state_dict() if hasattr(self.model_state, 'module') else self.model_state.state_dict()
        ilap_dict = self.model_ilap.module.state_dict() if hasattr(self.model_ilap, 'module') else self.model_ilap.state_dict()
        
        torch.save(state_dict, self.ckpt_dir / name_state)
        torch.save(ilap_dict, self.ckpt_dir / name_ilap)

    def run(self):
        best_loss = float('inf')
        
        for epoch in range(self.train_cfg['num_epochs']):
            if self.is_master:
                logging.info(f"Starting Epoch {epoch + 1}")
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            # test_loss = self.test()
            
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