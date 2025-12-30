import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

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

SAVE_PATH = '/home/csn801/MICCAI2025_benchmark/mistake_prediction/rf_with_id/'

COLUMNS_TO_DROP = [
    'acceleration_mean', 'acceleration_std', 'case', 'fname',
    'win_w', 'win_h', 'x', 'y', 'Unnamed: 0',
    'speed_mean', 'speed_std', 'steps_back',
    'timestamp', 'rad_name', 'gaze_path', 'radiologist'
]

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 250,
    'max_depth': 9,
    'random_state': 10
}


def preprocess(df):
    """Drop unnecessary columns from the dataframe."""
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    return df.drop(columns=cols_to_drop)


def has_only_nan_coordinates(df):
    """Check if 'x' and 'y' columns contain only NaN values."""
    x_all_nan = df['x'].isna().all() if 'x' in df.columns else True
    y_all_nan = df['y'].isna().all() if 'y' in df.columns else True
    return x_all_nan and y_all_nan


def load_dataset(input_name):
    """Load and concatenate data from all root paths."""
    dataframes = []
    sample_ids = []  # Case folder paths (e.g., .../9ddc082081a79f748fc6e16fcf52e1ac.png)
    
    total_files = 0
    invalid_files = 0
    invalid_file_paths = []
    
    for root_path in ROOT_PATHS:
        if not os.path.isdir(root_path):
            print(f"Warning: Root path not found: {root_path}")
            continue
            
        folders = [name for name in os.listdir(root_path) 
                   if os.path.isdir(os.path.join(root_path, name))]
        
        for folder_name in folders:
            # Case folder path (the sample identifier)
            case_folder = os.path.join(root_path, folder_name)
            # Full path to the feature CSV
            input_path = os.path.join(case_folder, 'miccai2025', input_name)
            
            if os.path.exists(input_path):
                total_files += 1
                df = pd.read_csv(input_path).tail(1)
                
                if df.empty or has_only_nan_coordinates(df):
                    invalid_files += 1
                    invalid_file_paths.append(input_path)
                    continue
                
                df = preprocess(df)
                dataframes.append(df)
                sample_ids.append(case_folder)  # Use case folder as sample ID
            else:
                invalid_files += 1
                invalid_file_paths.append(input_path)
                continue
    
    if total_files > 0:
        print(f"  Invalid files (empty or x/y all NaN): {invalid_files}/{total_files} ({100*invalid_files/total_files:.1f}%)")
    if invalid_file_paths and len(invalid_file_paths) <= 10:
        print(f"  Invalid file paths:")
        for path in invalid_file_paths:
            print(f"    - {path}")
    
    if not dataframes:
        raise ValueError(f"No data found for {input_name}")
    
    dataset = pd.concat(dataframes, ignore_index=True)
    
    # Track which rows are valid BEFORE dropping NaN
    valid_mask = ~dataset.isna().any(axis=1)
    
    # Filter sample_ids to match valid rows
    sample_ids_filtered = [sid for sid, valid in zip(sample_ids, valid_mask) if valid]
    
    dataset = dataset.dropna().reset_index(drop=True)
    
    return dataset, sample_ids_filtered


def train_and_evaluate(X, y, feature_names, sample_ids, input_name):
    """Train Random Forest with 5-fold CV and return predictions with sample IDs."""
    
    # Store predictions with their sample identifiers
    all_predictions = []
    importances = []
    fold_aucs = []
    
    split_log_path = os.path.join(SAVE_PATH, f'splits_{input_name.replace(".csv", ".txt")}')
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    with open(split_log_path, 'w') as f:
        f.write(f"Random Forest Parameters: {RF_PARAMS}\n\n")
        
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X), 1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Log the split
            f.write(f"FOLD {fold_idx}\n")
            f.write("TRAIN\n")
            train_ids = [sample_ids[idx] for idx in train_index]
            f.write(",".join(train_ids) + "\n")
            f.write("TEST\n")
            test_ids = [sample_ids[idx] for idx in test_index]
            f.write(",".join(test_ids) + "\n\n")

            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(X_train, y_train)

            importances.append(model.feature_importances_)
            probs = model.predict_proba(X_test)[:, 1]
            
            # Store predictions with sample IDs
            for idx, prob in zip(test_index, probs):
                all_predictions.append({
                    'sample_id': sample_ids[idx],
                    'y_pred': prob,
                    'y': y.iloc[idx]
                })
            
            # Calculate fold AUC
            fold_auc = roc_auc_score(y_test, probs)
            fold_aucs.append(fold_auc)

    print(f"  Split information saved to: {split_log_path}")
    print(f"  Per-fold AUCs: {[f'{auc:.3f}' for auc in fold_aucs]}")
    print(f"  Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    mean_importances = np.mean(importances, axis=0)
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importances
    }).sort_values(by='importance', ascending=False)
    
    # Create answers DataFrame with sample IDs
    answers_df = pd.DataFrame(all_predictions)
    
    return feat_imp_df, answers_df, fold_aucs


def main(input_name):
    """Main pipeline for a single input file."""
    print(f"Processing: {input_name}")
    
    dataset, sample_ids = load_dataset(input_name)
    print(f"  Loaded {len(dataset)} valid samples")
    
    # Check label distribution
    label_counts = dataset['label'].value_counts()
    print(f"  Label distribution: {dict(label_counts)}")
    
    y = dataset['label']
    X = dataset.drop(['label'], axis=1)
    print(f"  Features ({len(X.columns)}): {list(X.columns)}")
    
    feat_imp_df, answers_df, fold_aucs = train_and_evaluate(X, y, X.columns, sample_ids, input_name)
    
    feat_imp_df.to_csv(os.path.join(SAVE_PATH, f'feat_importance_{input_name}'), index=False)
    answers_df.to_csv(os.path.join(SAVE_PATH, f'predict_rad_{input_name}'), index=False)
    
    print(f"  Results saved to {SAVE_PATH}")
    print(f"  Prediction file includes {len(answers_df)} samples with IDs")
    
    # Show example sample_id format
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
    
    print("Random Forest 5-Fold CV - Mistake Prediction")
    print("=" * 70)
    print(f"Save path: {SAVE_PATH}")
    print(f"RF Parameters: {RF_PARAMS}")
    print(f"Feature files: {len(names)}\n")
    
    for input_name in names:
        try:
            main(input_name)
        except Exception as e:
            print(f"Error processing {input_name}: {e}\n")
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)