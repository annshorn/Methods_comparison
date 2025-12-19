import os
import pandas as pd
import json
import ray

ray.init(num_cpus=4)

base_path = "/home/csn801/__allData/locked_gaze_data/400"
doc_mistakes_400_path = "/home/csn801/__allData/400_doctor_mistake.csv"

doc_mistakes_400 = pd.read_csv(doc_mistakes_400_path).drop(columns=["Unnamed: 0"])
doc_mistakes_400 = doc_mistakes_400.drop_duplicates()
doc_mistakes_400_ref = ray.put(doc_mistakes_400)

SAVE_FOLDER = 'benchmarks'


@ray.remote
def process_folder(base_path, folder_name, doc_mistakes):
    """
    Preprocess the raw data.
    """
    folder_path = os.path.join(base_path, folder_name)
    for entry in os.listdir(folder_path):
        gaze_path = os.path.join(folder_path, entry, 'gaze.csv')
        if os.path.exists(gaze_path):
            gaze_df = pd.read_csv(gaze_path)
            meta_path = os.path.join(folder_path, entry, 'meta.json')
            if not os.path.exists(meta_path):
                print(f"Meta file missing for {folder_name}, {entry}")
                continue
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {meta_path}: {e}")
                continue
            # Extract info about radiologist, fname,
            # global_ind, num_part, num_img, x0, y0	
            # win_width, win_height
            meta_df = pd.DataFrame([meta])

            # Take only valid gaze points
            gaze_df = gaze_df[gaze_df['gaze_point_validity'].astype(str).isin(['1', '1.0', 'True'])]
            if gaze_df.empty:
                print(f"No valid gaze points for {folder_name}, {entry}")
                continue

            new_gaze_df = gaze_df[['x', 'y', 'timestamp', 'gaze_point_validity']].copy()
            new_gaze_df = new_gaze_df.assign(case='400')

            fields_to_copy = ['radiologist', 'fname', 'x0', 'y0', 'win_width', 'win_height']
            values = meta_df.loc[0, fields_to_copy]
            new_gaze_df = new_gaze_df.assign(**values.to_dict())

            new_gaze_df['x'] -= new_gaze_df['x0']
            new_gaze_df['y'] -= new_gaze_df['y0']

            matched_df = pd.merge(doc_mistakes, meta_df, 
                      on=['fname', 'global_ind', 'num_part', 'num_img'],
                      how='inner')
            
            if len(matched_df) > 1:
                raise ValueError(f"Multiple entries found in doc_mistakes_400 for {folder_name}, {entry}")
            
            if matched_df.empty:
                print(f"No doctor_mistake match for {folder_name}, {entry}")
                continue
            
            doctor_mistake_value = matched_df['doctor_mistake'].iloc[0]
            unique_values = matched_df['doctor_mistake'].nunique(dropna=True)
            
            if pd.isna(doctor_mistake_value) or unique_values != 1:
                print(f"Skipping {folder_name}, {entry}: invalid doctor_mistake (empty or multiple unique values)")
                continue

            new_gaze_df['doctor_mistake'] = doctor_mistake_value

            new_folder_path = os.path.join(folder_path, entry, SAVE_FOLDER)
            os.makedirs(new_folder_path, exist_ok=True)
            new_gaze_path = os.path.join(new_folder_path, 'preprocessed_raw_gaze.csv')
            new_gaze_df.to_csv(new_gaze_path, index=False)
            print(f"Processed: {folder_name}, {entry}")
        else:
            print(f"Gaze file does not exist for {folder_name}, {entry}")
    

folders_names = [entry for entry in os.listdir(base_path)
           if os.path.isdir(os.path.join(base_path, entry))]

futures = [process_folder.remote(base_path, folder_name, doc_mistakes_400_ref) 
           for folder_name in folders_names]

ray.get(futures)
ray.shutdown()

import glob
expected = len(glob.glob(f"{base_path}/**/gaze.csv", recursive=True))
actual = len(glob.glob(f"{base_path}/**/{SAVE_FOLDER}/preprocessed_raw_gaze.csv", recursive=True))
print(f"\n=== Verification ===")
print(f"Expected: {expected}, Processed: {actual}, Missing: {expected - actual}")