import os
import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_gaze_type_percentages(root_paths, file_names):
    """
    Calculate the percentage of each gaze type across all files.
    Also validates that EYE_MOVEMENT_TYPE is integer and calculates
    distance/time metrics between fixations and smooth pursuits.
    """
    results = {name: defaultdict(int) for name in file_names}
    total_counts = {name: 0 for name in file_names}
    
    # For tracking non-integer types
    non_integer_files = {name: [] for name in file_names}
    
    # For distance/time calculations between fixations (0) and smooth pursuits (2)
    all_distances = {name: [] for name in file_names}
    all_time_diffs = {name: [] for name in file_names}
    
    for root_path in root_paths:
        if not os.path.exists(root_path):
            print(f"Warning: Path does not exist: {root_path}")
            continue
            
        folders = [name for name in os.listdir(root_path) 
                   if os.path.isdir(os.path.join(root_path, name))]
        
        for folder_name in folders:
            for file_name in file_names:
                input_path = os.path.join(root_path, folder_name, 'benchmarks', file_name)
                
                if not os.path.exists(input_path):
                    continue
                
                try:
                    df = pd.read_csv(input_path)
                    if 'EYE_MOVEMENT_TYPE' not in df.columns:
                        print(f"Warning: 'EYE_MOVEMENT_TYPE' not found in {input_path}")
                        continue
                    
                    # 1. Check if all EYE_MOVEMENT_TYPE values are integers
                    if not pd.api.types.is_integer_dtype(df['EYE_MOVEMENT_TYPE']):
                        # Check if values are float but actually integers
                        if df['EYE_MOVEMENT_TYPE'].dropna().apply(lambda x: float(x).is_integer()).all():
                            df['EYE_MOVEMENT_TYPE'] = df['EYE_MOVEMENT_TYPE'].astype(int)
                        else:
                            non_integer_files[file_name].append(input_path)
                            print(f"Warning: Non-integer EYE_MOVEMENT_TYPE in {input_path}")
                            print(f"  Unique values: {df['EYE_MOVEMENT_TYPE'].unique()}")
                    
                    counts = df['EYE_MOVEMENT_TYPE'].value_counts()
                    for gaze_type, count in counts.items():
                        results[file_name][gaze_type] += count
                    total_counts[file_name] += len(df)
                    
                    # 2. Calculate distance and time between fixations (0) and smooth pursuits (2)
                    distances, time_diffs = calculate_fixation_pursuit_metrics(df)
                    all_distances[file_name].extend(distances)
                    all_time_diffs[file_name].extend(time_diffs)
                    
                except Exception as e:
                    print(f"Error reading {input_path}: {e}")
    
    # Convert counts to percentages
    percentages = {}
    for file_name in file_names:
        if total_counts[file_name] > 0:
            percentages[file_name] = {
                gaze_type: (count / total_counts[file_name]) * 100
                for gaze_type, count in results[file_name].items()
            }
            percentages[file_name]['_total_samples'] = total_counts[file_name]
        else:
            percentages[file_name] = {'_total_samples': 0}
        
        # Add non-integer file info
        percentages[file_name]['_non_integer_files'] = non_integer_files[file_name]
        
        # Add distance/time metrics
        if all_distances[file_name]:
            percentages[file_name]['_mean_distance_fix_pursuit'] = np.mean(all_distances[file_name])
            percentages[file_name]['_std_distance_fix_pursuit'] = np.std(all_distances[file_name])
        if all_time_diffs[file_name]:
            percentages[file_name]['_min_time_fix_pursuit'] = np.min(all_time_diffs[file_name])
            percentages[file_name]['_mean_time_fix_pursuit'] = np.mean(all_time_diffs[file_name])
    
    return percentages


def calculate_fixation_pursuit_metrics(df):
    """
    Calculate distances and time differences between fixation points (0) 
    and smooth pursuit points (2).
    
    Returns lists of distances and time differences for transitions.
    """
    distances = []
    time_diffs = []
    
    # Need x, y coordinates and time columns
    # Adjust column names based on your data structure
    x_col = 'x' if 'x' in df.columns else 'X' if 'X' in df.columns else None
    y_col = 'y' if 'y' in df.columns else 'Y' if 'Y' in df.columns else None
    time_col = 'timestamp' if 'timestamp' in df.columns else 't' if 't' in df.columns else None
    
    if x_col is None or y_col is None:
        return distances, time_diffs
    
    # Find transitions between fixation (0) and smooth pursuit (2)
    df = df.reset_index(drop=True)
    
    for i in range(1, len(df)):
        curr_type = df.loc[i, 'EYE_MOVEMENT_TYPE']
        prev_type = df.loc[i-1, 'EYE_MOVEMENT_TYPE']
        
        # Check for transition between fixation (0) and smooth pursuit (2) in either direction
        if (curr_type == 0 and prev_type == 2) or (curr_type == 2 and prev_type == 0):
            # Calculate Euclidean distance
            x1, y1 = df.loc[i-1, x_col], df.loc[i-1, y_col]
            x2, y2 = df.loc[i, x_col], df.loc[i, y_col]
            
            if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append(dist)
            
            # Calculate time difference
            if time_col and time_col in df.columns:
                t1, t2 = df.loc[i-1, time_col], df.loc[i, time_col]
                if pd.notna(t1) and pd.notna(t2):
                    time_diffs.append(abs(t2 - t1))
    
    return distances, time_diffs


def print_gaze_percentages(percentages):
    """Pretty print the gaze type percentages and metrics."""
    gaze_type_names = {0: 'Fixation', 1: 'Saccade', 2: 'Smooth Pursuit', 3: 'Blinks/Noise'}
    
    for file_name, gaze_data in percentages.items():
        print(f"\n{'='*60}")
        print(f"File: {file_name}")
        print(f"Total samples: {gaze_data.get('_total_samples', 0):,}")
        print("-" * 40)
        
        print("Gaze Type Percentages:")
        for gaze_type, pct in sorted(gaze_data.items(), key=lambda x: str(x[0])):
            if not str(gaze_type).startswith('_'):
                type_name = gaze_type_names.get(gaze_type, f'Type {gaze_type}')
                print(f"  {gaze_type} ({type_name}): {pct:.2f}%")
        
        print("-" * 40)
        print("Data Type Validation:")
        non_int = gaze_data.get('_non_integer_files', [])
        if non_int:
            print(f"  WARNING: {len(non_int)} files with non-integer EYE_MOVEMENT_TYPE")
        else:
            print("  ✓ All EYE_MOVEMENT_TYPE values are integers")
        
        print("-" * 40)
        print("Fixation ↔ Smooth Pursuit Metrics:")
        if '_mean_distance_fix_pursuit' in gaze_data:
            print(f"  Mean distance: {gaze_data['_mean_distance_fix_pursuit']:.4f}")
            print(f"  Std distance:  {gaze_data['_std_distance_fix_pursuit']:.4f}")
        else:
            print("  No distance data available (check x/y columns)")
            
        if '_min_time_fix_pursuit' in gaze_data:
            print(f"  Min time diff: {gaze_data['_min_time_fix_pursuit']:.4f}")
            print(f"  Mean time diff: {gaze_data['_mean_time_fix_pursuit']:.4f}")
        else:
            print("  No time data available (check time column)")


# Usage
if __name__ == "__main__":
    root_paths = [
        "/home/csn801/__allData/locked_gaze_data/400/06.06.2021",
        "/home/csn801/__allData/locked_gaze_data/400/05.06.2021",
        "/home/csn801/__allData/locked_gaze_data/400/10.04.2021",
        "/home/csn801/__allData/locked_gaze_data/400/14.11.2021",
        "/home/csn801/__allData/locked_gaze_data/600/02.04.2022",
        "/home/csn801/__allData/locked_gaze_data/600/12.03.2022",
        "/home/csn801/__allData/locked_gaze_data/600/19.03.2022",
        "/home/csn801/__allData/locked_gaze_data/600/27.03.2022"
    ]
    
    file_names = [
        'cnn_lstm_model_gazecom_BATCH-8192_EPOCHS-25_movement_type.csv'
    ]
    
    percentages = calculate_gaze_type_percentages(root_paths, file_names)
    print_gaze_percentages(percentages)