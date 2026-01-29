import numpy as np
import h5py
import torch
import torch.utils.data as data
import logging
import yaml
import os
from pathlib import Path
from typing import Optional

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _env_or_default(env_key: str, default: Optional[str]) -> Optional[str]:
    """Helper to read a path-like value from environment variables."""
    val = os.environ.get(env_key)
    return val if val not in (None, "") else default

def get_training_config(config_path='config.yaml'):
    """
    Helper to read the training configuration section from yaml.
    Returns the dictionary corresponding to trainings[selected_model].
    """
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        if not os.path.exists(config_path):
            config_path = '../config.yaml' 
            
    try:
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        selected_model = full_config.get('selected_model', 'HOM')
        return full_config['trainings'][selected_model]
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}. Using defaults. Error: {e}")
        return {}

class GLORYSDataLoader:
    """Singleton Data Manager to handle file handles efficiently across workers."""
    
    _instance = None
    _file_handles = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GLORYSDataLoader, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_file_handles(cls, years, base_path, forcing_type='NeuralOM', forcing_path=None):
        """
        Lazy loads H5 file handles for the requested years.
        Supports loading auxiliary forcing files if forcing_type is 'WenHai'.
        """
        # Include forcing config in the cache key to avoid mixing different settings.
        key = (tuple(years), forcing_type)
        
        if key not in cls._file_handles:
            cls._file_handles[key] = {}
            for year in years:
                try:
                    # 1. Load Main GLORYS Data
                    pc_file = Path(base_path) / f'GLORYS_05_{year}.h5'
                    if not pc_file.exists():
                        raise FileNotFoundError(f"Data file not found: {pc_file}")

                    cls._file_handles[key][f'pc_{year}'] = h5py.File(pc_file, 'r')['data']
                    
                    # 2. Load Auxiliary Forcing Data (If WenHai mode)
                    if forcing_type == 'WenHai':
                        if forcing_path is None:
                            # Default fallback path if not provided in config
                            forcing_path = _env_or_default("HOM_ERA5_MEAN_SURFACE_DIR", None)
                            if forcing_path is None:
                                raise ValueError(
                                    "forcing_type='WenHai' requires forcing_path in config, "
                                    "or set env HOM_ERA5_MEAN_SURFACE_DIR."
                                )
                        
                        f_file = Path(forcing_path) / f'WenHai_forcing_05_{year}.h5'
                        if not f_file.exists():
                            raise FileNotFoundError(f"Forcing file not found: {f_file}")
                            
                        # Store with a specific key prefix
                        cls._file_handles[key][f'forcing_{year}'] = h5py.File(f_file, 'r')['data']
                        logger.info(f"Loaded forcing data (WenHai) for year {year}")

                    logger.info(f"Loaded GLORYS data for year {year}")
                except Exception as e:
                    logger.error(f"Failed to load year {year}: {e}")
                    raise
        return cls._file_handles[key]

class GLORYSBaseDataset(data.Dataset):
    """
    Base Dataset class supporting both random and fixed-interval sampling.
    """
    
    def __init__(self, years, sampling_mode='fixed', day_interval=3, num_samples_per_year=120,
                 ds_factor=1, lat_range=(0, 720), lon_range=(0, 1440), time_range=1, base_path=None):
        super(GLORYSBaseDataset, self).__init__()
        
        self.years = years
        self.sampling_mode = sampling_mode # 'random' or 'fixed'
        self.day_interval = day_interval
        self.num_samples_per_year = num_samples_per_year
        self.time_range = time_range
        self.ds_factor = ds_factor
        self.lat_start, self.lat_end = lat_range
        self.lon_start, self.lon_end = lon_range
        
        self.base_path = base_path or _env_or_default("HOM_GLORYS_05_H5_DIR", None)
        if self.base_path is None:
            raise ValueError(
                "GLORYS base_path is not set. Pass base_path=... or set env HOM_GLORYS_05_H5_DIR "
                "to the directory containing files like GLORYS_05_<YEAR>.h5."
            )
        
        # --- Config Loading for Forcing Logic ---
        config = get_training_config()
        self.forcing_type = config.get('forcing_type', 'NeuralOM') # Default to old logic
        self.forcing_path = config.get('forcing_path', _env_or_default("HOM_ERA5_MEAN_SURFACE_DIR", None))
        
        logger.info(f"Dataset initialized with forcing_type: {self.forcing_type}")

        # Preload handles (Updated to pass forcing config)
        self.file_handles = GLORYSDataLoader.get_file_handles(
            self.years, self.base_path, self.forcing_type, self.forcing_path
        )

        # Generate indices based on mode
        self.day_indices = self._generate_day_indices()

        # --- Load Climate Mean Fields ---
        self.use_mean_field = config.get('mean_field', False) 
        self.mean_file_path = _env_or_default("HOM_CLIMATE_MEAN_NPY", None)
        
        self.thetao_mean_full, self.so_mean_full, self.zos_mean_full = None, None, None

        if self.use_mean_field:
            if self.mean_file_path is None:
                raise ValueError(
                    "mean_field=True requires a climate mean npy file. "
                    "Set env HOM_CLIMATE_MEAN_NPY to the path of climate_mean_s_t_ssh.npy."
                )
            try:
                logger.info(f"Loading climate mean fields from {self.mean_file_path}...")
                raw_mean_data = np.load(self.mean_file_path)
                self.thetao_mean_full = np.flip(raw_mean_data[:, 69:92:1, 1:, :], 2)
                self.so_mean_full = np.flip(raw_mean_data[:, 0:23:1, 1:, :], 2)
                self.zos_mean_full = np.flip(raw_mean_data[:, 92, 1:, :], 1)
            except Exception as e:
                logger.error(f"Failed to load climate mean fields: {e}")
                raise e

    def _generate_day_indices(self):
        """Generates indices based on the sampling mode."""
        day_indices = []
        for year in self.years:
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            days_in_year = 366 if is_leap else 365
            last_valid_day = days_in_year - self.time_range
            
            if self.sampling_mode == 'random':
                # Random sampling mode (for Training)
                available_range = np.arange(0, last_valid_day)
                n_to_sample = min(self.num_samples_per_year, len(available_range))
                sampled_days = np.random.choice(available_range, size=n_to_sample, replace=False)
                for d in sampled_days:
                    day_indices.append((year, int(d)))
            else:
                # Fixed interval mode (for Validation/Test)
                for day_of_year in range(0, last_valid_day, self.day_interval):
                    day_indices.append((year, day_of_year))
        
        # Shuffle indices to mix years within a batch
        # np.random.shuffle(day_indices)
        return day_indices

    def reshuffle_train_samples(self):
        """Force a refresh of random samples. Only effective if mode is 'random'."""
        if self.sampling_mode == 'random':
            self.day_indices = self._generate_day_indices()
            logger.info(f"Dataset samples reshuffled. New count: {len(self.day_indices)}")

    def _read_data_slice(self, year, day_of_year, variable, lat_slice, lon_slice):
        t_start, t_end = day_of_year, day_of_year + self.time_range + 1
        
        # --- New Logic for Atmospheric Forcing (Variable 'A') ---
        if variable == 'A' and self.forcing_type == 'WenHai':
            # Use the new forcing handle
            handle = self.file_handles[f'forcing_{year}']
            
            # The new file has shape 365*8*360*720
            # Warning: If year is leap year (366 days), verify if forcing file is 365 or 366.
            # Assuming standard calendar alignment.
            
            # Channel selection:
            # The original 'A' (NeuralOM) uses 3 channels. The new file has 8.
            # To maintain compatibility with existing models (simvp, etc.), we slice the first 3.
            # Modify `slice(0, 3)` below if you need specific channels from the 8 available.
            forcing_channel_slice = slice(0, 8) 
            
            data = handle[t_start:t_end, forcing_channel_slice, lat_slice, lon_slice]
            # No zero-masking logic specified for new forcing, assuming raw data is correct.
            return np.nan_to_num(data, nan=0.0)

        # --- Original Logic for all other variables (and 'A' if forcing_type == 'NeuralOM') ---
        handle = self.file_handles[f'pc_{year}']
        
        var_map = {
                'uo': (slice(0, 23), True), 'vo': (slice(23, 46), True), 'zos': (46, True),
                'thetao': (slice(47, 70), True), 'so': (slice(70, 93), True),
                'sic': (93, False), 'sit': (94, False), 'usi': (95, False),
                'vsi': (96, False), 'A': (slice(97, 100), False)
        }
        
        idx, apply_mask = var_map[variable]
        data = handle[t_start:t_end, idx, lat_slice, lon_slice]
        if apply_mask: data[..., -10:, :] = 0
        return np.nan_to_num(data, nan=0.0)
    
    def __getitem__(self, index):
        year, day_of_year = self.day_indices[index]
        lat_slice = slice(self.lat_start, self.lat_end, self.ds_factor)
        lon_slice = slice(self.lon_start, self.lon_end, self.ds_factor)
        
        vars_to_load = ['thetao', 'so', 'uo', 'vo', 'zos', 'sic', 'sit', 'usi', 'vsi', 'A']
        tensors = [torch.tensor(self._read_data_slice(year, day_of_year, v, lat_slice, lon_slice), 
                                dtype=torch.float32) for v in vars_to_load]
            
        if self.use_mean_field:
            m_idx = [(day_of_year + t) % 365 for t in range(self.time_range + 1)]
            tensors.append(torch.tensor(self.thetao_mean_full[m_idx, ..., lat_slice, lon_slice], dtype=torch.float32))
            tensors.append(torch.tensor(self.so_mean_full[m_idx, ..., lat_slice, lon_slice], dtype=torch.float32))
            tensors.append(torch.tensor(self.zos_mean_full[m_idx, lat_slice, lon_slice], dtype=torch.float32))
        else:
            tensors.extend([torch.zeros((1, 1, 1, 1), dtype=torch.float32), 
                            torch.zeros((1, 1, 1, 1), dtype=torch.float32), 
                            torch.zeros((1, 1, 1), dtype=torch.float32)])
        
        return tuple(tensors)
    
    def __len__(self):
        return len(self.day_indices)

# --- Subclasses with Specific Sampling Behaviors ---

class TrainDataset(GLORYSBaseDataset):
    def __init__(self, ds_factor=1, lat_range=(0, 360), lon_range=(0, 720)):
        config = get_training_config()
        super(TrainDataset, self).__init__(
            years=range(1993, 2019),
            sampling_mode='fixed',
            day_interval=3,  # ~120 samples/year
            ds_factor=ds_factor, 
            lat_range=lat_range, 
            lon_range=lon_range, 
            time_range=config.get('integral_interval', 1)
        )

class ValDataset(GLORYSBaseDataset):
    def __init__(self, ds_factor=1, lat_range=(0, 360), lon_range=(0, 720)):
        config = get_training_config()
        super(ValDataset, self).__init__(
            years=range(2019, 2020),
            sampling_mode='fixed',
            day_interval=6,  # sample every 6 days
            ds_factor=ds_factor, 
            lat_range=lat_range, 
            lon_range=lon_range, 
            time_range=config.get('integral_interval', 1)
        )
        
class TestDataset(GLORYSBaseDataset):
    def __init__(self, ds_factor=1, lat_range=(0, 360), lon_range=(0, 720)):
        config = get_training_config()
        super(TestDataset, self).__init__(
            years=range(2020, 2021),
            sampling_mode='fixed',
            day_interval=30,  # sample every 30 days
            ds_factor=ds_factor, 
            lat_range=lat_range, 
            lon_range=lon_range, 
            time_range=config.get('integral_interval_test', 60)
        )
