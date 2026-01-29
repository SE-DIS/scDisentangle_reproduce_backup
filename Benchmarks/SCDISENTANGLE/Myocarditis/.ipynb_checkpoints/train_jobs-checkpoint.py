import os
import sys
import shutil
from multiprocessing import Pool

import yaml

sys.path.append("../")
import train_scdisentangle

yaml_path = "configs/myocarditis.yaml"
max_epochs = 521

custom_name = yaml_path.split("/")[-1].split(".")[0]
data_name = "Myocarditis"

donors = ['SIC_264', 'SIC_171', 'SIC_258', 'SIC_177', 'SIC_48', 'SIC_232']

def run_single_job(args):
    """
    Single training job for one OOD and seed on a given GPU device.
    """
    ood_cov, seed_nb, device = args

    save_root = f"weights/{custom_name}/{seed_nb}"
    job_dir = f"{save_root}/{ood_cov}"

    # Skip if already processed
    if os.path.exists(job_dir):
        print(ood_cov, seed_nb, "already processed, skipping")
        return

    print("processing", ood_cov, "seed", seed_nb, "on device", device)

    with open(yaml_path, "r") as stream:
        hparams = yaml.safe_load(stream)

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(job_dir, exist_ok=True)

    # Configure OOD timepoint and experiment bookkeeping
    hparams["OOD"]["filter_dict"]["donor"] = ood_cov
    hparams["wandb"]["name"] += f"{ood_cov}_{seed_nb}"
    hparams["train"]["set_seed"] = seed_nb
    hparams["growing_neurons"]["prior_mappers"]["mappers"]["donor_mapper"][
        "collapse_name"
    ] = ood_cov
    hparams["evaluations"]["context_transfer_criterion"]["kwargs"][
        "ood_covariate"
    ] = ood_cov

    hparams["hardware"]["device"] = device
    hparams["save_experiment"]["experiment_path"] = f"{job_dir}/"

    shutil.copy2(yaml_path, job_dir)
    with open(f"{job_dir}/ood.txt", "w") as f:
        f.write(ood_cov + "\n")
    
    train_scdisentangle.make_run(
        hparams,
        max_epochs,
        seed_nb=seed_nb,
        save_path=f"{job_dir}/",
        split_key=None,
    )
if __name__ == "__main__":
    seed_nbs = list(range(1, 11)) #[42]

    # Distribute ALL OOD covariates across GPUs, with up to 6 concurrent jobs total
    # (3 per GPU on average). Jobs are queued globally: when one finishes, the next
    # ood_cov starts, assigned alternately to GPU 0 and 1.

    jobs = []
    for i1, cov in enumerate(donors):
        for i2, seed_nb in enumerate(seed_nbs):
            device = (i1+i2) % 2  # even indices -> GPU 0, odd indices -> GPU 1
            jobs.append((cov, seed_nb, device))

    print("Queued jobs (cov, device):", jobs)
    
    with Pool(processes=4) as pool:
        pool.map(run_single_job, jobs)