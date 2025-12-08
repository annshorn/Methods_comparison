import features
import os
import ray
import scipy
import pandas as pd
import matplotlib.image as mpimg
import warnings
import numpy as np
import logging
from sklearn.cluster import HDBSCAN

warnings.filterwarnings("ignore")
np.seterr(all='ignore')
logging.getLogger("ray").setLevel(logging.ERROR)


NUM_WORKERS = 80
BASE_PATHS = ["/home/csn801/EyeTrackProject/data/400",
              "/home/csn801/EyeTrackProject/data/600"]

IMAGE_FOLDER = '/home/csn801/__allData/converted_images'
MASK_FOLDER = '/home/csn801/__allData/segmentation_masks/all_masks_machine'

NAMES = {
    "input_name": [
        'imst_mst.csv',
        'tobii_ivt.csv',
        'idt.csv',
        'ihmm.csv',
        'cnn_blstm_model_hmr_HDBSCAN_fix.csv',
        'tcn_model_gazecom_HDBSCAN_fix.csv',
        'tcn_model_hmr_HDBSCAN_fix.csv',
        'cnn_lstm_model_hmr_HDBSCAN_fix.csv',
        'sp_tool_all_movement_type.csv',
    ],
    "output_name": [
        'imst_mst_features.csv',
        'tobii_ivt_features.csv',
        'idt_features.csv',
        'ihmm_features.csv',
        'cnn_blstm_model_hmr_HDBSCAN_fix_features.csv',
        'tcn_model_gazecom_HDBSCAN_fix_features.csv',
        'tcn_model_hmr_HDBSCAN_fix_features.csv',
        'cnn_lstm_model_hmr_HDBSCAN_fix_features.csv',
        'sp_tool_features.csv',
    ]
}

FEATURE_NAMES = [
    'fixation_dist', 'fixation_dist_std',
    'fixation_angle', 'fixation_angle_std',
    '#_fixations', 'switches_between_objects', 'total_length',
    'info_gain_per_fixation_mean', 'info_gain_per_fixation_std',
    'acceleration_mean', 'acceleration_std',
    '#_fixation_below_75pix', '#_fixation_below_150pix',
    'visits_lung_L', 'visits_lung_R',
    'gaze_coverage_abs_L', 'gaze_coverage_L',
    'gaze_coverage_abs_R', 'gaze_coverage_R',
    '#_fixations_infogain_below_50pix',
    '#_fixations_infogain_below_100pix',
    '#_fixations_infogain_below_200pix',
    '#_fixations_infogain_below_300pix',
    '#_fixations_infogain_below_5000pix',
    '%_fixations_infogain_below_50pix',
    '%_fixations_infogain_below_100pix',
    '%_fixations_infogain_below_200pix',
    '%_fixations_infogain_below_300pix',
    '%_fixations_infogain_below_5000pix'
]

HDBSCAN_FILES = [
    'cnn_blstm_model_hmr_HDBSCAN_fix.csv',
    'tcn_model_gazecom_HDBSCAN_fix.csv',
    'tcn_model_hmr_HDBSCAN_fix.csv',
    'cnn_lstm_model_hmr_HDBSCAN_fix.csv',
    '1DCNNBLSTM_out.csv'
]


def perform_dbscan_and_find_centers(points, min_cluster_size=2, min_samples=None):
    """
    Perform HDBSCAN clustering on points and calculate the centroids of each cluster.
    """
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusterer.fit(points)
    
    labels = clusterer.labels_
    unique_labels = set(labels)
    centroids = []

    for label in unique_labels:
        if label != -1:  # Ignore noise points
            members = points[labels == label]
            centroid = members.mean(axis=0)
            centroids.append(centroid)

    return np.array(centroids), labels


def chunkify(lst, n):
    """Divide a list `lst` into `n` chunks."""
    return [lst[i::n] for i in range(n)]


@ray.remote
def process_cases(cases, name_fixpoints, name_output):
    """Process a list of (folder_path, image_folder) cases."""
    for folder_path, entry in cases:
        fix_points_path = os.path.join(folder_path, entry, 'miccai2025', name_fixpoints)
        features_outpath = os.path.join(folder_path, entry, 'miccai2025', name_output)
        
        if not os.path.exists(fix_points_path):
            continue
            
        try:
            fix_points_df = pd.read_csv(fix_points_path)
            
            if name_fixpoints == 'imst_mst.csv':
                fix_points_df = fix_points_df[(fix_points_df['x'] >= 0) & (fix_points_df['y'] >= 0)].reset_index(drop=True)
                first_row_values = fix_points_df.iloc[0]
                fix_points_df = fix_points_df.fillna(first_row_values)
                
            if name_fixpoints == 'sp_tool_all_movement_type.csv':
                fix_points_df = fix_points_df[fix_points_df['EYE_MOVEMENT_TYPE'] == 'FIX'].reset_index(drop=True)
                fix_points_df['time'] = fix_points_df['time'] / 1000000
                fix_points_df.rename(columns={'time': 'timestamp'}, inplace=True)

                raw_gaze = pd.read_csv(os.path.join(folder_path, entry, 'miccai2025', 'raw_gaze.csv'))
                fix_points_df['fname'] = raw_gaze['fname'].iloc[0]
                fix_points_df['case'] = raw_gaze['case'].iloc[0]
                fix_points_df['radiologist'] = raw_gaze['radiologist'].iloc[0]
                fix_points_df['win_w'] = raw_gaze['win_w'].iloc[0]
                fix_points_df['win_h'] = raw_gaze['win_h'].iloc[0]
                fix_points_df['label'] = raw_gaze['label'].iloc[0]

            if name_fixpoints in HDBSCAN_FILES:
                if len(fix_points_df) > 250:
                    centroids, _ = perform_dbscan_and_find_centers(fix_points_df[['x', 'y']].values)
                    centroids_df = pd.DataFrame(centroids, columns=['x', 'y'])
                    centroids_df['case'] = fix_points_df['case'].iloc[0]
                    centroids_df['radiologist'] = fix_points_df['radiologist'].iloc[0]
                    centroids_df['fname'] = fix_points_df['fname'].iloc[0]
                    centroids_df['win_w'] = fix_points_df['win_w'].iloc[0]
                    centroids_df['win_h'] = fix_points_df['win_h'].iloc[0]
                    centroids_df['label'] = fix_points_df['label'].iloc[0]
                    fix_points_df = centroids_df

            fname = fix_points_df['fname'].iloc[0]

            image = mpimg.imread(os.path.join(IMAGE_FOLDER, fname))
            mask_L = mpimg.imread(os.path.join(MASK_FOLDER, fname[:-4] + '_left_lung.png'))
            mask_L = scipy.ndimage.zoom(mask_L, image.shape[0] / mask_L.shape[0], order=0)
            mask_R = mpimg.imread(os.path.join(MASK_FOLDER, fname[:-4] + '_right_lung.png'))
            mask_R = scipy.ndimage.zoom(mask_R, image.shape[0] / mask_R.shape[0], order=0)

            seq = []
            for i in range(len(fix_points_df)):
                x_seq = fix_points_df[['x', 'y']].loc[:i].values
                stats = features.gaze_statistics_case(x_seq, cut_off=75)
                seq.append(stats.get_all_statistics(image, mask_L, mask_R, x_seq))

            df = pd.DataFrame(seq, columns=FEATURE_NAMES)
            df['x'] = fix_points_df['x']
            df['y'] = fix_points_df['y']
            columns_to_copy = ['case', 'radiologist', 'fname', 'win_w', 'win_h', 'label']
            values = fix_points_df[columns_to_copy].iloc[0]
            df = df.assign(**values.to_dict())
            df.to_csv(features_outpath, index=False)
            print(f"Preprocessed: {fix_points_path}")
            
        except Exception as e:
            print(f"Error processing {fix_points_path}: {e}, {len(fix_points_df)} rows")


def get_all_cases():
    """Get all (date_folder, image_folder) pairs from base paths."""
    all_cases = []
    for base_path in BASE_PATHS:
        for date_entry in os.listdir(base_path):
            date_folder = os.path.join(base_path, date_entry)
            if os.path.isdir(date_folder):
                for image_entry in os.listdir(date_folder):
                    image_folder = os.path.join(date_folder, image_entry)
                    if os.path.isdir(image_folder):
                        all_cases.append((date_folder, image_entry))
    return all_cases


def process_all_files(inp_name, otp_name, all_cases):
    """Process all cases for a given input/output file pair."""
    print(f"\n{'='*60}")
    print(f"Processing: {inp_name} -> {otp_name}")
    print(f"Total cases: {len(all_cases)}")
    print(f"{'='*60}")
    
    case_chunks = chunkify(all_cases, NUM_WORKERS)
    ray.get([process_cases.remote(chunk,
                                  name_fixpoints=inp_name,
                                  name_output=otp_name) for chunk in case_chunks])
    
    print(f"Completed: {inp_name} -> {otp_name}")


def main():
    ray.init(num_cpus=NUM_WORKERS, logging_level=logging.ERROR)
    
    try:
        all_cases = get_all_cases()
        print(f"Found {len(all_cases)} total cases to process")
        
        for inp_name, otp_name in zip(NAMES["input_name"], NAMES["output_name"]):
            process_all_files(inp_name, otp_name, all_cases)
        
        print("\n" + "="*60)
        print("All processing completed!")
        print("="*60)
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()