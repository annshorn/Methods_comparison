import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
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

SAVE_PATH = '/home/csn801/MICCAI2025_benchmark/expertise_est/rf_5kfold_final/'

COLUMNS_TO_DROP = [
    'acceleration_mean', 'acceleration_std', 'case', 'fname',
    'win_w', 'win_h', 'label', 'x', 'y', 'Unnamed: 0',
    'speed_mean', 'speed_std', 'steps_back',
    'timestamp', 'rad_name', 'gaze_path'
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
    path_dont_exist = 0
    paths_dont_exist = []
    
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
                print('PATH DOESNT EXIST:', input_path)
                path_dont_exist += 1
                paths_dont_exist.append(input_path)
                continue
    
    print(f"  Invalid files (empty or x/y all NaN): {invalid_files}/{total_files} ({100*invalid_files/total_files:.1f}%)")
    if invalid_file_paths:
        print(f"  Invalid file paths:", len(invalid_file_paths))
        for path in invalid_file_paths:
            print(f"    - {path}")

    if paths_dont_exist:
        print(f"  Paths dont exist:", len(paths_dont_exist))
        for path in paths_dont_exist:
            print(f"    - {path}")
    
    if not dataframes:
        raise ValueError(f"No data found for {input_name}")
    
    dataset = pd.concat(dataframes, ignore_index=True)
    
    valid_mask = ~dataset.isna().any(axis=1)
    file_paths = [fp for fp, valid in zip(file_paths, valid_mask) if valid]
    
    dataset = dataset.dropna().reset_index(drop=True)
    dataset['label'] = dataset['radiologist'].map({'a': 0, 'b': 0, 'c': 0, 'd': 1})
    
    return dataset, file_paths


def train_and_evaluate(X, y, feature_names, file_paths, input_name):
    """Train Random Forest with 5-fold CV and return predictions and importances."""
    y_probs = []
    y_true = []
    importances = []
    
    split_log_path = os.path.join(SAVE_PATH, f'splits_{input_name.replace(".csv", ".txt")}')
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    with open(split_log_path, 'w') as f:
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X), 1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Log the split - comma separated file paths only
            f.write(f"FOLD {fold_idx}\n")
            f.write("TRAIN\n")
            train_paths = [file_paths[idx] for idx in train_index]
            f.write(",".join(train_paths) + "\n")
            f.write("TEST\n")
            test_paths = [file_paths[idx] for idx in test_index]
            f.write(",".join(test_paths) + "\n\n")

            model = RandomForestClassifier(max_depth=5, random_state=10)
            model.fit(X_train, y_train)

            importances.append(model.feature_importances_)
            y_probs.extend(model.predict_proba(X_test)[:, 1])
            y_true.extend(y_test)

    print(f"  Split information saved to: {split_log_path}")

    mean_importances = np.mean(importances, axis=0)
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importances
    }).sort_values(by='importance', ascending=False)
    
    answers_df = pd.DataFrame({'y_pred': y_probs, 'y': y_true})
    
    return feat_imp_df, answers_df


def main(input_name):
    """Main pipeline for a single input file."""
    print(f"Processing: {input_name}")
    
    dataset, file_paths = load_dataset(input_name)
    print(f"  Loaded {len(dataset)} valid samples")
    
    y = dataset['label']
    X = dataset.drop(['radiologist', 'label'], axis=1)
    print(f"  Features: {list(X.columns)}")
    
    feat_imp_df, answers_df = train_and_evaluate(X, y, X.columns, file_paths, input_name)
    
    feat_imp_df.to_csv(os.path.join(SAVE_PATH, f'feat_importance_{input_name}'), index=False)
    answers_df.to_csv(os.path.join(SAVE_PATH, f'predict_rad_{input_name}'), index=False)
    
    print(f"  Results saved to {SAVE_PATH}\n")


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
            print(f"Error processing {input_name}: {e}\n")
