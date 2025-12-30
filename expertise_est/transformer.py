import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import os
import math
import torch
import torch.nn as nn

ROOT_PATHS = [
    "/home/csn801/EyeTrackProject/data/400/06.06.2021",
    "/home/csn801/EyeTrackProject/data/400/05.06.2021",
    "/home/csn801/EyeTrackProject/data/400/10.04.2021",
    "/home/csn801/EyeTrackProject/data/400/14.11.2021",
    "/home/csn801/EyeTrackProject/data/600/02.04.2022",
    "/home/csn801/EyeTrackProject/data/600/12.03.2022",
    "/home/csn801/EyeTrackProject/data/600/19.03.2022",
    "/home/csn801/EyeTrackProject/data/600/27.03.2022"
]

SAVE_PATH = '/home/csn801/MICCAI2025_benchmark/expertise_est/transformer_5kfolds_new/'

COLUMNS_TO_DROP = [
    'acceleration_mean', 'acceleration_std', 'case', 'fname',
    'win_w', 'win_h', 'label', 'x', 'y', 'Unnamed: 0',
    'speed_mean', 'speed_std', 'steps_back',
    'timestamp', 'rad_name', 'gaze_path'
]

RADIOLOGIST_MAPPING = {'a': 0, 'b': 0, 'c': 0, 'd': 1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(df):
    """Drop unnecessary columns from the dataframe."""
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    return df.drop(columns=cols_to_drop)


def has_only_nan_coordinates(df):
    """Check if 'x' and 'y' columns contain only NaN values."""
    x_all_nan = df['x'].isna().all() if 'x' in df.columns else True
    y_all_nan = df['y'].isna().all() if 'y' in df.columns else True
    return x_all_nan and y_all_nan


def map_radiologist_label(df, sample_id):
    """Map radiologist to label with case-insensitive matching and validation."""
    # Convert to lowercase for case-insensitive matching
    df['radiologist_lower'] = df['radiologist'].astype(str).str.lower().str.strip()
    df['label'] = df['radiologist_lower'].map(RADIOLOGIST_MAPPING)
    
    # Check for unmapped values
    if df['label'].isna().any():
        unmapped = df[df['label'].isna()]['radiologist'].unique()
        print(f"  Warning: Unmapped radiologist values in {sample_id}: {unmapped}")
        return None
    
    df = df.drop(columns=['radiologist_lower'])
    return df


def create_dict_dataset(input_name):
    """Load data and return dict with sample_id as key."""
    dataset = {}
    
    total_files = 0
    invalid_files = 0
    unmapped_files = 0
    
    for root_path in ROOT_PATHS:
        if not os.path.isdir(root_path):
            print(f"Warning: Root path not found: {root_path}")
            continue
            
        folders = [name for name in os.listdir(root_path) 
                   if os.path.isdir(os.path.join(root_path, name))]
        
        for folder_name in folders:
            case_folder = os.path.join(root_path, folder_name)
            input_path = os.path.join(case_folder, 'miccai2025', input_name)
            
            if os.path.exists(input_path):
                total_files += 1
                df = pd.read_csv(input_path)
                
                # Check for invalid data before dropping NaN
                if df.empty or has_only_nan_coordinates(df):
                    invalid_files += 1
                    continue
                
                df = df.dropna().reset_index(drop=True)
                
                if df.empty:
                    invalid_files += 1
                    continue
                
                # Validate radiologist mapping before adding to dataset
                if 'radiologist' in df.columns:
                    unique_rads = df['radiologist'].astype(str).str.lower().str.strip().unique()
                    invalid_rads = [r for r in unique_rads if r not in RADIOLOGIST_MAPPING]
                    if invalid_rads:
                        print(f"  Skipping {case_folder}: invalid radiologist values {invalid_rads}")
                        unmapped_files += 1
                        continue
                
                df = preprocess(df)
                dataset[case_folder] = df
    
    print(f"  Loaded {len(dataset)} valid samples")
    print(f"  Invalid files: {invalid_files}/{total_files}")
    if unmapped_files > 0:
        print(f"  Skipped due to unmapped radiologists: {unmapped_files}")
    
    return dataset


def split_dataset_into_folds(dataset, n_splits=5):
    """Split dataset into folds, preserving sample_ids."""
    sample_ids = list(dataset.keys())
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    folds = {}
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(sample_ids)):
        train_ids = [sample_ids[i] for i in train_idx]
        val_ids = [sample_ids[i] for i in val_idx]
        
        folds[fold_idx] = {
            'train': {sid: dataset[sid] for sid in train_ids},
            'val': {sid: dataset[sid] for sid in val_ids}
        }
    
    return folds


def return_reading_seq(seq, sequence_length):
    """Pad or truncate sequence to fixed length."""
    if len(seq) < sequence_length:
        pad_length = sequence_length - len(seq)
        seq = torch.cat((seq, torch.full((pad_length, seq.shape[1]), -1, dtype=torch.float32)))
    elif len(seq) > sequence_length:
        seq = seq[-sequence_length:]
    return seq


class TDataset(torch.utils.data.Dataset):
    """Dataset that preserves sample_ids for later alignment."""
    
    def __init__(self, grouped_data, sequence_length):
        self.sequences = []
        self.labels = []
        self.sample_ids = []
        
        for sample_id, data in grouped_data.items():
            data = data.reset_index(drop=True)
            
            # Case-insensitive radiologist mapping
            data['radiologist_lower'] = data['radiologist'].astype(str).str.lower().str.strip()
            data['label'] = data['radiologist_lower'].map(RADIOLOGIST_MAPPING)
            
            # Skip if any NaN labels (should not happen after validation, but safety check)
            if data['label'].isna().any():
                print(f"  Warning: Skipping {sample_id} due to NaN labels")
                continue
            
            seq = data.drop(['radiologist', 'radiologist_lower', 'label'], axis=1).values
            
            # Check for NaN/Inf in sequence data
            if np.isnan(seq).any() or np.isinf(seq).any():
                print(f"  Warning: Skipping {sample_id} due to NaN/Inf in features")
                continue
            
            seq = return_reading_seq(torch.tensor(seq, dtype=torch.float32), sequence_length)
            
            self.sequences.append(seq)
            self.labels.append(int(data['label'].iloc[0]))
            self.sample_ids.append(sample_id)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], idx


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self,
                 input_size: int,
                 d_model: int = 512,
                 nhead: int = 4,
                 num_layers: int = 2,
                 output_size: int = 2,
                 max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Add input normalization for numerical stability
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, output_size)
        self.sf = torch.nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create mask for padded positions (padded with -1)
        padding_mask = (x[:, :, 0] == -1)
        
        # Replace padding values with 0 for computation
        x = torch.where(x == -1, torch.zeros_like(x), x)
        
        # Normalize input
        x = self.input_norm(x)
        
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = x[:, -1, :]  # Use the last time step
        logits = self.output_proj(x)
        return self.sf(logits)[:, 1]


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    
    for sequences, labels, _ in train_loader:
        optimizer.zero_grad()
        
        sequences = sequences.to(device)
        labels = labels.float().to(device)
        
        outputs = model(sequences)
        
        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            print("  Warning: NaN detected in model outputs during training, skipping batch")
            continue
        
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss):
            print("  Warning: NaN loss detected, skipping batch")
            continue
        
        loss.backward()
        
        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, val_dataset, device):
    """Validate and return predictions with sample IDs."""
    model.eval()
    predictions = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels, indices in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = outputs.cpu().numpy()
            
            # Check for NaN in predictions
            if np.isnan(probs).any():
                print("  Warning: NaN detected in validation predictions")
                # Replace NaN with 0.5 (neutral prediction)
                probs = np.nan_to_num(probs, nan=0.5)
            
            for i, idx in enumerate(indices):
                sample_id = val_dataset.sample_ids[idx]
                predictions.append({
                    'sample_id': sample_id,
                    'y_pred': probs[i],
                    'y': labels[i].item()
                })
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    return predictions, np.array(all_probs), np.array(all_labels)


def main(input_name):
    """Main training pipeline with sample_id tracking."""
    print(f"Processing: {input_name}")
    
    dataset = create_dict_dataset(input_name)
    
    if len(dataset) == 0:
        print(f"  Error: No valid samples loaded for {input_name}")
        return None
    
    folds = split_dataset_into_folds(dataset, n_splits=5)
    
    all_predictions = []
    fold_aucs = []
    
    for fold_idx, fold in folds.items():
        print(f"  Fold {fold_idx + 1}/5")
        
        train_dataset = TDataset(fold['train'], sequence_length=100)
        val_dataset = TDataset(fold['val'], sequence_length=100)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"    Warning: Empty dataset in fold {fold_idx + 1}, skipping")
            continue
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False
        )
        
        input_size = train_dataset.sequences[0].shape[1]
        
        model = Model(input_size)
        model = model.to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        
        # Training with early stopping based on loss
        num_epochs = 200
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break
        
        # Validation
        predictions, probs, labels = validate(model, val_loader, val_dataset, device)
        all_predictions.extend(predictions)
        
        # Calculate fold AUC with safety check
        try:
            fold_auc = roc_auc_score(labels, probs)
            fold_aucs.append(fold_auc)
            print(f"    Fold {fold_idx + 1} AUC: {fold_auc:.4f}")
        except ValueError as e:
            print(f"    Error calculating AUC for fold {fold_idx + 1}: {e}")
            print(f"    Labels unique: {np.unique(labels)}, Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
    
    if len(fold_aucs) > 0:
        print(f"  Per-fold AUCs: {[f'{auc:.3f}' for auc in fold_aucs]}")
        print(f"  Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    
    # Save results
    answers_df = pd.DataFrame(all_predictions)
    output_filename = f'predict_rad_{input_name}'
    answers_df.to_csv(os.path.join(SAVE_PATH, output_filename), index=False)
    
    print(f"  Results saved to {SAVE_PATH}")
    print(f"  Prediction file includes {len(answers_df)} samples with IDs")
    
    if len(answers_df) > 0:
        print(f"  Example sample_id: {answers_df['sample_id'].iloc[0]}")
    print()
    
    return fold_aucs


if __name__ == "__main__":
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    names = [
        'imst_mst_features.csv',
        'ihmm_features.csv',
        'idt_features.csv',
        'tobii_ivt_features.csv',
        'sp_tool_features.csv',
        '1DCNNBLSTM_HDBSCAN_features.csv',
        'tcn_model_gazecom_HDBSCAN_fix_features.csv',
        'tcn_model_hmr_HDBSCAN_fix_features.csv',
        'cnn_lstm_model_hmr_HDBSCAN_fix_features.csv',
        'cnn_blstm_model_hmr_HDBSCAN_fix_features.csv',
    ]
    
    for input_name in names:
        try:
            main(input_name)
        except Exception as e:
            import traceback
            print(f"Error processing {input_name}: {e}")
            traceback.print_exc()
            print()
