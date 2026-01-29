import os
import sys
import shutil
from multiprocessing import Pool

import yaml
import json

import train_scdisentangle_norman

yaml_path = "configs/norman.yaml"
max_epochs = 521

custom_name = 'single_only'
data_name = "Norman"

with open(f'../../../Datasets/preprocessed_datasets/norman_splits_2.json') as f:
    _split_dict = json.load(f)

run_indices = list(_split_dict.keys())

def run_single_job(args):
    """
    Single training job for one OOD and seed on a given GPU device.
    """
    split_idx, seed_nb, device = args
    
    split_dict = _split_dict[split_idx]
    
    save_root = f"weights/{custom_name}/{seed_nb}"
    job_dir = f"{save_root}/{split_idx}"
    
    # Skip if already processed
    if os.path.exists(job_dir):
        print(split_idx, seed_nb, "already processed, skipping")
        return
    
    print("processing", split_idx, "seed", seed_nb, "on device", device)
    
    with open(yaml_path, "r") as stream:
        hparams = yaml.safe_load(stream)
    
    hparams['wandb']['group'] = 'single_only'
    
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(job_dir, exist_ok=True)
    
    # Configure OOD timepoint and experiment bookkeeping
    hparams["wandb"]["name"] += f"{split_idx}_{seed_nb}"
    hparams["train"]["set_seed"] = seed_nb
    
    hparams["hardware"]["device"] = device
    hparams["save_experiment"]["experiment_path"] = f"{job_dir}/"
    
    shutil.copy2(yaml_path, job_dir)
    with open(f"{job_dir}/ood.txt", "w") as f:
        f.write(split_idx + "\n")
    
    train_scdisentangle_norman.make_run(
        hparams,
        max_epochs,
        seed_nb=seed_nb,
        save_path=f"{job_dir}/",
        split_dict=split_dict,
    )
    
if __name__ == "__main__":
    # Use the same seed as in train.py
    seed_nbs = [42]
    
    # Distribute ALL OOD covariates across GPUs, with up to 6 concurrent jobs total
    # (3 per GPU on average). Jobs are queued globally: when one finishes, the next
    # ood_cov starts, assigned alternately to GPU 0 and 1.
    
    jobs = []
    for i1, split_idx in enumerate(run_indices):
        for i2, seed_nb in enumerate(seed_nbs):
            device = (i1+i2) % 2  # even indices -> GPU 0, odd indices -> GPU 1
            jobs.append((split_idx, seed_nb, device))
    
    print("Queued jobs (cov, device):", jobs)
    
    with Pool(processes=2) as pool:
        pool.map(run_single_job, jobs)