import ray
import os

ray.init()

num_workers = int(ray.available_resources()['CPU'])
print(f"Number of workers: {num_workers}")

def chunkify(lst, n):
    """Divide a list `lst` into `n` chunks."""
    return [lst[i::n] for i in range(n)]

@ray.remote
def process_arff_files(folder_list):
    target = "The number after speed, direction denotes the step size that was used for the calculation."
    for folder_path in folder_list:
        input_filename = os.path.join(folder_path, 'benchmarks', 'raw_1DCNNBLSTM.arff')
        if os.path.exists(input_filename):
            with open(input_filename, 'r') as file:
                metadata_lines = []
                modified_lines = []
                
                for line in file:
                    if target in line:
                        continue
                    elif line.strip().startswith('% %@METADATA'):
                        metadata_line = line.replace('% ', '', 1)
                        metadata_lines.append(metadata_line)
                    elif line.strip().startswith('@RELATION'):
                        relation_line = '@RELATION gaze_labels\n'
                        modified_lines.append(relation_line)
                        modified_lines.extend(metadata_lines)
                        # modified_lines.append('%@METADATA distance_mm 725\n')
                    else:
                        modified_lines.append(line)
            
            with open(input_filename, 'w') as file:
                file.writelines(modified_lines)
            print(f">>> processed {input_filename}")
        else:
            print(f">>> skipping {input_filename} (not found)")

if __name__ == "__main__":
    basepaths = [
        "/home/csn801/__allData/locked_gaze_data/400",
        "/home/csn801/__allData/locked_gaze_data/600",
    ]
    
    all_folders = []
    for basepath in basepaths:
        for date_folder in os.listdir(basepath):
            date_path = os.path.join(basepath, date_folder)
            if os.path.isdir(date_path):
                for hash_folder in os.listdir(date_path):
                    hash_path = os.path.join(date_path, hash_folder)
                    if os.path.isdir(hash_path):
                        all_folders.append(hash_path)
    
    print(f"Found {len(all_folders)} folders to process")
    
    chunks = chunkify(all_folders, num_workers)
    
    futures = [process_arff_files.remote(chunk) for chunk in chunks if chunk]
    
    print(f"Launched {len(futures)} tasks across {num_workers} workers")
    
    ray.get(futures)
    
    print("All done!")
    ray.shutdown()