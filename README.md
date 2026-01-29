# HybridOM (paper code release)

This repository contains the official training/inference code used in our paper, focusing on:

- **0.5° global experiment**: `exp_main_05/`
- **regional high-resolution experiment**: `exp_spatial_highres/`

All hard-coded personal/cluster paths have been removed for open-sourcing. Please configure data locations via **environment variables** and/or per-experiment `config.yaml`.

## Abstract

Global ocean modeling is vital for climate science but struggles to balance computational efficiency with accuracy. Traditional numerical solvers are accurate but computationally expensive, while pure deep learning approaches, though fast, often lack physical consistency and long-term stability. To address this, we introduce HybridOM, a framework integrating a lightweight, differentiable numerical solver as a skeleton to enforce physical laws, with a neural network as the flesh to correct subgrid-scale dynamics. To enable efficient high-resolution modeling, we further introduce a physics-informed regional downscaling mechanism based on flux gating. This design achieves the inference efficiency of AI-based methods while preserving the accuracy and robustness of physical models. Extensive experiments on the GLORYS12V1 and OceanBench dataset validate HybridOM's performance in two distinct regimes: long-term subseasonal-to-seasonal simulation and short-term operational forecasting coupled with the FuXi-2.0 weather model. Results demonstrate that HybridOM achieves state-of-the-art accuracy while strictly maintaining physical consistency, offering a robust solution for next-generation ocean digital twins.

## Requirements

- Python + PyTorch with `torch.distributed` (NCCL recommended)
- For single-node multi-GPU DDP, use `torchrun`

## Data path configuration (required)

The code reads the following environment variables (and will error out if required ones are missing):

- **`HOM_GLORYS_05_H5_DIR`**: global GLORYS directory containing `GLORYS_05_<YEAR>.h5`
- **`HOM_GLORYS_REGIONAL_025_H5_DIR`**: regional directory containing `GLORYS_pc_025_<YEAR>.h5` (regional experiment)
- **`HOM_REGIONAL_MERGED_RESULTS_DIR`**: optional merged forecast directory used by the regional dataloader
- **`HOM_ERA5_MEAN_SURFACE_DIR`**: required when `forcing_type: 'WenHai'`, containing `WenHai_forcing_05_<YEAR>.h5`
- **`HOM_CLIMATE_MEAN_NPY`**: required when `mean_field: true`, pointing to `climate_mean_s_t_ssh.npy`

`mask_path`, `mask_ori_path`, and related items are configured in each experiment’s `config.yaml` (defaulting to `./data/...` under that experiment folder).

## Run with DDP

### 0.5° global experiment (`exp_main_05/`)

Train:

```bash
export HOM_GLORYS_05_H5_DIR="/path/to/GLORYS/05/h5_dir"
# Optional:
# export HOM_ERA5_MEAN_SURFACE_DIR="/path/to/ERA5/mean_surface"
# export HOM_CLIMATE_MEAN_NPY="/path/to/climate_mean_s_t_ssh.npy"

cd exp_main_05
torchrun --standalone --nproc_per_node 8 train_hom.py
```

Inference:

```bash
export HOM_GLORYS_05_H5_DIR="/path/to/GLORYS/05/h5_dir"

cd exp_main_05
torchrun --standalone --nproc_per_node 8 inference_hom.py
```

### Regional high-resolution experiment (`exp_spatial_highres/`)

Train:

```bash
export HOM_GLORYS_05_H5_DIR="/path/to/GLORYS/05/h5_dir"
export HOM_GLORYS_REGIONAL_025_H5_DIR="/path/to/GLORYS/regional_025/h5_dir"
# Optional:
# export HOM_REGIONAL_MERGED_RESULTS_DIR="/path/to/merged_results_dir"
# export HOM_CLIMATE_MEAN_NPY="/path/to/climate_mean_s_t_ssh.npy"

cd exp_spatial_highres
torchrun --standalone --nproc_per_node 8 train_hom.py
```

Inference:

```bash
export HOM_GLORYS_05_H5_DIR="/path/to/GLORYS/05/h5_dir"
export HOM_GLORYS_REGIONAL_025_H5_DIR="/path/to/GLORYS/regional_025/h5_dir"

cd exp_spatial_highres
torchrun --standalone --nproc_per_node 8 inference_hom.py
```

