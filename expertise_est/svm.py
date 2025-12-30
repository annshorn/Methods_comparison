import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import os
import sys

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

SAVE_PATH = '/home/csn801/MICCAI2025_benchmark/expertise_est/svm_5kfold_final/'

COLUMNS_TO_DROP = [
    'acceleration_mean', 'acceleration_std', 'case', 'fname',
    'win_w', 'win_h', 'label', 'x', 'y', 'Unnamed: 0',
    'speed_mean', 'speed_std', 'steps_back',
    'timestamp', 'rad_name', 'gaze_path'
]

# SVM configurations: (name, params)
SVM_CONFIGS = [
    ('SVC_linear', {'kernel': 'linear', 'C': 1}),
    ('SVC_rbf', {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'}),
    ('SVC_poly_d2', {'kernel': 'poly', 'C': 1, 'degree': 2}),
    ('SVC_poly_d3', {'kernel': 'poly', 'C': 1, 'degree': 3}),
    ('SVC_sigmoid', {'kernel': 'sigmoid', 'C': 1}),
]

NAMES = [
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
    file_paths = []
    
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
            input_path = os.path.join(root_path, folder_name, 'miccai2025', input_name)
            if os.path.exists(input_path):
                total_files += 1
                df = pd.read_csv(input_path).tail(1)
                
                if df.empty or has_only_nan_coordinates(df):
                    invalid_files += 1
                    invalid_file_paths.append(input_path)
                    continue
                
                df = preprocess(df)
                dataframes.append(df)
                file_paths.append(input_path)
            else:
                invalid_files += 1
                invalid_file_paths.append(input_path)
                continue
    
    if total_files > 0:
        print(f"  Invalid files (empty or x/y all NaN): {invalid_files}/{total_files} ({100*invalid_files/total_files:.1f}%)")
    if invalid_file_paths:
        print(f"  Invalid file paths:")
        for path in invalid_file_paths:
            print(f"    - {path}")
    
    if not dataframes:
        raise ValueError(f"No data found for {input_name}")
    
    dataset = pd.concat(dataframes, ignore_index=True)
    
    valid_mask = ~dataset.isna().any(axis=1)
    file_paths = [fp for fp, valid in zip(file_paths, valid_mask) if valid]
    
    dataset = dataset.dropna().reset_index(drop=True)
    dataset['label'] = dataset['radiologist'].map({'a': 0, 'b': 0, 'c': 0, 'd': 1})
    
    return dataset, file_paths


def get_feature_importance(model, X_test, y_test, kernel):
    """
    Get feature importance for SVM.
    - Linear kernel: use absolute coefficients
    - Other kernels: use permutation importance
    """
    if kernel == 'linear':
        # Linear SVM has coef_ attribute
        return np.abs(model.coef_[0])
    else:
        # Use permutation importance for non-linear kernels
        perm_imp = permutation_importance(
            model, X_test, y_test, 
            n_repeats=10, 
            random_state=42,
            scoring='roc_auc'
        )
        return perm_imp.importances_mean


def train_and_evaluate_svm(X, y, feature_names, file_paths, input_name, svm_name, svm_params):
    """Train SVM with 5-fold CV and return predictions and feature importances."""
    y_probs = []
    y_true = []
    importances = []
    
    split_log_path = os.path.join(SAVE_PATH, f'splits_{svm_name}_{input_name.replace(".csv", ".txt")}')
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    with open(split_log_path, 'w') as f:
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X), 1):
            X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Log the split - comma separated file paths only
            f.write(f"FOLD {fold_idx}\n")
            f.write("TRAIN\n")
            train_paths = [file_paths[idx] for idx in train_index]
            f.write(",".join(train_paths) + "\n")
            f.write("TEST\n")
            test_paths = [file_paths[idx] for idx in test_index]
            f.write(",".join(test_paths) + "\n\n")

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train SVM
            model = SVC(**svm_params, probability=True)
            model.fit(X_train_scaled, y_train)
            
            # Get feature importance
            fold_importance = get_feature_importance(
                model, X_test_scaled, y_test, svm_params['kernel']
            )
            importances.append(fold_importance)
            
            # Get predictions
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            y_probs.extend(y_prob)
            y_true.extend(y_test)

    # Calculate mean feature importance across folds
    mean_importances = np.mean(importances, axis=0)
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importances
    }).sort_values(by='importance', ascending=False)
    
    answers_df = pd.DataFrame({'y_pred': y_probs, 'y': y_true})
    
    return feat_imp_df, answers_df


def main(input_name):
    """Main pipeline for a single input file."""
    print(f"\nProcessing: {input_name}")
    print("=" * 70)
    
    dataset, file_paths = load_dataset(input_name)
    print(f"  Loaded {len(dataset)} valid samples")
    
    y = dataset['label']
    X = dataset.drop(['radiologist', 'label'], axis=1)
    print(f"  Features ({len(X.columns)}): {list(X.columns)}")
    
    # Train and evaluate each SVM configuration
    for svm_name, svm_params in SVM_CONFIGS:
        print(f"\n  Training {svm_name}...")
        
        feat_imp_df, answers_df = train_and_evaluate_svm(
            X, y, X.columns, file_paths, input_name, svm_name, svm_params
        )
        
        # Save predictions
        pred_path = os.path.join(SAVE_PATH, f'predict_{svm_name}_{input_name}')
        answers_df.to_csv(pred_path, index=False)
        print(f"    Predictions saved: {pred_path}")
        
        # Save feature importance
        imp_path = os.path.join(SAVE_PATH, f'feat_importance_{svm_name}_{input_name}')
        feat_imp_df.to_csv(imp_path, index=False)
        print(f"    Feature importance saved: {imp_path}")
    
    print(f"\n  Completed all SVM configurations for {input_name}")


if __name__ == "__main__":
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    print("SVM 5-Fold Cross-Validation Training")
    print("=" * 70)
    print(f"Save path: {SAVE_PATH}")
    print(f"SVM configurations: {[name for name, _ in SVM_CONFIGS]}")
    print(f"Feature files: {len(NAMES)}")
    
    for input_name in NAMES:
        try:
            main(input_name)
        except Exception as e:
            print(f"Error processing {input_name}: {e}\n")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
