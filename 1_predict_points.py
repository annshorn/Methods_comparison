"""
Eye Movement Type Classification using CNN-LSTM Ensemble (GPU-enabled)

This script processes raw gaze data and classifies eye movements using
a pre-trained CNN-LSTM ensemble with majority voting.
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import mode

import ray
from tqdm import tqdm

# Local imports
from cnn_lstm import CNN_LSTM
import preprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('movement_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the movement classification pipeline."""
    # Model parameters
    input_size: int = 25
    output_size: int = 4
    kernel_size: int = 5
    dropout: float = 0.25
    features: int = 20
    lstm_layers: int = 2 
    bidirectional: bool = False
    
    # Processing parameters
    timesteps: int = 25
    batch_size: int = 512
    frequency: int = 100
    window_length: int = 1
    stride: int = 10
    
    # Parallelization
    num_cpus: int = 25
    num_gpus: int = 1  # Number of GPUs available
    workers_per_gpu: int = 2  # How many workers share each GPU
    use_gpu: bool = True  # Toggle GPU usage
    
    # Output naming
    output_filename: str = 'cnn_lstm_model_gazecom_BATCH-8192_EPOCHS-25_movement_type.csv'


@dataclass
class Paths:
    """Path configuration."""
    model_dir: Path = Path('/home/csn801/BENCHMARK/1DCNN-LSTM-GazeCom/final_models')
    model_pattern: str = 'cnn_lstm_model_gazecom_BATCH-8192_EPOCHS-25_FOLD-{fold}.pt'
    
    data_roots: Tuple[str, ...] = (
        "/home/csn801/__allData/locked_gaze_data/400/06.06.2021/",
        "/home/csn801/__allData/locked_gaze_data/400/05.06.2021/",
        "/home/csn801/__allData/locked_gaze_data/400/10.04.2021/",
        "/home/csn801/__allData/locked_gaze_data/400/14.11.2021/",
        "/home/csn801/__allData/locked_gaze_data/600/02.04.2022/",
        "/home/csn801/__allData/locked_gaze_data/600/12.03.2022/",
        "/home/csn801/__allData/locked_gaze_data/600/19.03.2022/",
        "/home/csn801/__allData/locked_gaze_data/600/27.03.2022/",
    )
    
    def get_model_paths(self, num_folds: int = 5) -> List[Path]:
        """Generate model paths for all folds."""
        return [
            self.model_dir / self.model_pattern.format(fold=i)
            for i in range(1, num_folds + 1)
        ]


class TimeSeriesDataset(Dataset):
    """Dataset for time series classification with sliding window."""
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, timesteps: int):
        self.X = X
        self.Y = Y
        self.timesteps = timesteps

    def __len__(self) -> int:
        return len(self.Y) - (self.timesteps - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx += self.timesteps - 1
        sample_X = self.X[idx - self.timesteps + 1:idx + 1, :]
        sample_Y = self.Y[idx]
        return (
            torch.from_numpy(sample_X).float(),
            torch.from_numpy(np.array(sample_Y)).long()
        )


class EnsemblePredictor:
    """Handles ensemble prediction with majority voting."""
    
    def __init__(self, config: Config, model_paths: List[Path] = None, 
                 state_dicts: List[dict] = None, device: torch.device = None):
        self.config = config
        
        # Determine device
        if device is not None:
            self.device = device
        elif config.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Load models from paths or state dicts
        if state_dicts is not None:
            self.models = self._load_from_state_dicts(state_dicts)
        elif model_paths is not None:
            self.models = self._load_models(model_paths)
        else:
            raise ValueError("Must provide either model_paths or state_dicts")
        
        # Move models to device
        for model in self.models:
            model.to(self.device)
    
    def _create_model(self) -> CNN_LSTM:
        """Create a new model instance with config parameters."""
        return CNN_LSTM(
            input_size=self.config.input_size,
            output_size=self.config.output_size,
            kernel_size=self.config.kernel_size,
            dropout=self.config.dropout,
            features=self.config.features,
            lstm_layers=self.config.lstm_layers,
            bidirectional=self.config.bidirectional
        )
    
    def _load_from_state_dicts(self, state_dicts: List[dict]) -> List[CNN_LSTM]:
        """Load models from state dictionaries."""
        models = []
        for state_dict in state_dicts:
            model = self._create_model()
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        return models
    
    def _load_models(self, model_paths: List[Path]) -> List[CNN_LSTM]:
        """Load all fold models from files."""
        models = []
        for path in model_paths:
            if not path.exists():
                logger.warning(f"Model not found: {path}")
                continue
            
            model = self._create_model()
            state_dict = torch.load(path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
            logger.info(f"Loaded model: {path.name}")
        
        if not models:
            raise ValueError("No models were loaded successfully")
        
        return models
    
    def _predict_single_model(
        self, 
        model: CNN_LSTM, 
        X: np.ndarray, 
        Y: np.ndarray
    ) -> torch.Tensor:
        """Run predictions for a single model."""
        dataset = TimeSeriesDataset(X, Y, self.config.timesteps)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=(self.device.type == 'cuda'),  # Faster GPU transfer
            num_workers=0  # Avoid multiprocessing issues in Ray
        )
        
        all_preds = []
        
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                output = model(X_batch)
                preds = output.argmax(dim=1)
                all_preds.append(preds.cpu())
        
        return torch.cat(all_preds, dim=0)
    
    def predict(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Run ensemble prediction with majority voting."""
        all_predictions = []
        
        for model in self.models:
            preds = self._predict_single_model(model, X, Y)
            all_predictions.append(preds.numpy())
        
        # Majority voting
        predictions_stack = np.stack(all_predictions, axis=0)
        majority_vote = mode(predictions_stack, axis=0, keepdims=False)
        return majority_vote.mode


def process_folder(
    folder_path: Path,
    predictor: EnsemblePredictor,
    pproc: preprocessor.Preprocessor,
    config: Config
) -> Optional[str]:
    """Process a single folder and save predictions."""
    rawgaze_path = folder_path / 'benchmarks' / 'preprocessed_raw_gaze.csv'
    output_path = folder_path / 'benchmarks' / config.output_filename
    
    # Skip if input doesn't exist or output already exists
    if not rawgaze_path.exists():
        return f"Skipped (no input): {folder_path.name}"
    
    if output_path.exists():
        return f"Skipped (exists): {folder_path.name}"
    
    try:
        # Load and preprocess data
        rawgaze = pd.read_csv(rawgaze_path)
        rawgaze.rename(columns={'gaze_point_validity': 'Confidence'}, inplace=True)
        rawgaze['Pattern'] = 3
        rawgaze['X_coord'] = rawgaze['x'] / rawgaze['win_width']
        rawgaze['Y_coord'] = rawgaze['y'] / rawgaze['win_height']
        
        data = rawgaze[['X_coord', 'Y_coord', 'Confidence', 'Pattern']]
        X, Y = pproc.process_data(data)
        
        # Get predictions
        predictions = predictor.predict(X, Y)
        
        # Create output dataframe
        skip_rows = int(np.ceil(config.frequency * config.window_length)) + (config.timesteps - 1)
        output_df = pd.DataFrame({
            'x': rawgaze['x'].iloc[skip_rows:].values,
            'y': rawgaze['y'].iloc[skip_rows:].values,
            'timestamp': rawgaze['timestamp'].iloc[skip_rows:].values,
            'EYE_MOVEMENT_TYPE': predictions
        })
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        
        return f"Processed: {folder_path.name}"
        
    except Exception as e:
        logger.error(f"Error processing {folder_path}: {e}")
        return f"Error: {folder_path.name} - {str(e)}"


def create_gpu_worker(num_gpus_per_worker: float):
    """
    Factory function to create Ray remote function with GPU resources.
    
    Args:
        num_gpus_per_worker: Fractional GPU allocation (e.g., 0.5 = 2 workers per GPU)
    """
    @ray.remote(num_gpus=num_gpus_per_worker)
    def process_folder_batch_gpu(
        folder_paths: List[str],
        model_state_dicts: List[dict],
        config_dict: dict
    ) -> List[str]:
        """Process a batch of folders using GPU."""
        config = Config(**config_dict)
        
        # Determine which GPU to use (Ray assigns CUDA_VISIBLE_DEVICES)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Worker using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.warning("GPU requested but not available, falling back to CPU")
        
        # Create predictor with GPU
        predictor = EnsemblePredictor(
            config=config,
            state_dicts=model_state_dicts,
            device=device
        )
        
        pproc = preprocessor.Preprocessor(
            window_length=config.window_length,
            offset=0,
            stride=config.stride,
            frequency=config.frequency
        )
        
        results = []
        for folder_path in folder_paths:
            result = process_folder(Path(folder_path), predictor, pproc, config)
            results.append(result)
        
        return results
    
    return process_folder_batch_gpu


@ray.remote
def process_folder_batch_cpu(
    folder_paths: List[str],
    model_state_dicts: List[dict],
    config_dict: dict
) -> List[str]:
    """Process a batch of folders using CPU only."""
    config = Config(**config_dict)
    
    predictor = EnsemblePredictor(
        config=config,
        state_dicts=model_state_dicts,
        device=torch.device('cpu')
    )
    
    pproc = preprocessor.Preprocessor(
        window_length=config.window_length,
        offset=0,
        stride=config.stride,
        frequency=config.frequency
    )
    
    results = []
    for folder_path in folder_paths:
        result = process_folder(Path(folder_path), predictor, pproc, config)
        results.append(result)
    
    return results


def chunkify(lst: List, n: int) -> List[List]:
    """Divide a list into n roughly equal chunks."""
    return [lst[i::n] for i in range(n)]


def main():
    """Main entry point."""
    config = Config()
    paths = Paths()
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    use_gpu = config.use_gpu and gpu_available
    
    if config.use_gpu and not gpu_available:
        logger.warning("GPU requested but not available. Falling back to CPU.")
    
    # Initialize Ray with appropriate resources
    if use_gpu:
        ray.init(num_cpus=config.num_cpus, num_gpus=config.num_gpus)
        num_workers = config.num_gpus * config.workers_per_gpu
        gpu_fraction = 1.0 / config.workers_per_gpu
        process_batch = create_gpu_worker(gpu_fraction)
        logger.info(f"Initialized Ray with {config.num_gpus} GPU(s), {num_workers} workers")
    else:
        ray.init(num_cpus=config.num_cpus)
        num_workers = config.num_cpus
        process_batch = process_folder_batch_cpu
        logger.info(f"Initialized Ray with {num_workers} CPU workers")
    
    try:
        # Load model state dicts once
        model_paths = paths.get_model_paths()
        model_state_dicts = []
        
        for path in model_paths:
            if path.exists():
                state_dict = torch.load(path, map_location='cpu')
                model_state_dicts.append(state_dict)
                logger.info(f"Loaded model state: {path.name}")
        
        if not model_state_dicts:
            raise ValueError("No models found")
        
        # Put model states in Ray object store
        model_states_ref = ray.put(model_state_dicts)
        
        config_dict = {
            'input_size': config.input_size,
            'output_size': config.output_size,
            'kernel_size': config.kernel_size,
            'dropout': config.dropout,
            'features': config.features,
            'lstm_layers': config.lstm_layers,
            'bidirectional': config.bidirectional,
            'timesteps': config.timesteps,
            'batch_size': config.batch_size,
            'frequency': config.frequency,
            'window_length': config.window_length,
            'stride': config.stride,
            'output_filename': config.output_filename,
            'use_gpu': use_gpu,
            'num_gpus': config.num_gpus,
            'workers_per_gpu': config.workers_per_gpu,
            'num_cpus': config.num_cpus,
        }
        
        # Process each data root
        for root_path in paths.data_roots:
            root = Path(root_path)
            if not root.exists():
                logger.warning(f"Root path not found: {root}")
                continue
            
            folders = [
                str(root / name)
                for name in os.listdir(root)
                if (root / name).is_dir()
            ]
            
            if not folders:
                logger.info(f"No folders in {root}")
                continue
            
            logger.info(f"Processing {len(folders)} folders in {root}")
            
            # Distribute work across workers
            folder_chunks = chunkify(folders, num_workers)
            
            # Submit tasks
            futures = [
                process_batch.remote(chunk, model_states_ref, config_dict)
                for chunk in folder_chunks
                if chunk
            ]
            
            # Collect results
            all_results = []
            for future in tqdm(futures, desc=f"Processing {root.name}"):
                results = ray.get(future)
                all_results.extend(results)
                for result in results:
                    if result:
                        logger.info(result)
            
            logger.info(f"Completed {root}: {len(all_results)} folders processed")
    
    finally:
        ray.shutdown()
        logger.info("Ray shutdown complete")


if __name__ == '__main__':
    main()