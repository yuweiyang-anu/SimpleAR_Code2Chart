import os
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

datasets = ["code2chart_sft"]
resolution = 1024
root = "/openseg_blob/v-yuweiyang/SimpleAR/datasets/visual_tokens/"
# total = 163189
total = 10

def check_file(i, code_dir, label_dir):
    code_path = os.path.join(code_dir, f"{i}.npy")
    label_path = os.path.join(label_dir, f"{i}.npy")
    if os.path.exists(code_path) and os.path.exists(label_path):
        return {"code_path": code_path, "label_path": label_path}
    return None  # Skip missing files

def process_dataset(dataset):
    code_dir = os.path.join(root, f"{dataset}/{resolution}_codes")
    label_dir = os.path.join(root, f"{dataset}/{resolution}_labels")
    
    # Use multiprocessing to speed up file existence checks
    with Pool(processes=cpu_count()) as pool:
        meta = list(filter(None, tqdm(pool.starmap(check_file, [(i, code_dir, label_dir) for i in range(total)]), total=total)))
    
    save_path = os.path.join(root, f"{dataset}_{resolution}_{len(meta)}_meta.json")
    with open(save_path, "w") as f:
        json.dump(meta, f, indent=4)
    
    print(f"Saved {len(meta)} metadata entries to {save_path}")

if __name__ == "__main__":
    for dataset in datasets:
        process_dataset(dataset)
