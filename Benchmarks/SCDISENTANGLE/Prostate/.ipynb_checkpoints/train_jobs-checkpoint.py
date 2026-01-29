import os
import sys
import shutil
from multiprocessing import Pool

import yaml

sys.path.append("../")
import train_scdisentangle  # noqa: E402


yaml_path = "configs/prostate.yaml"
max_epochs = 521

custom_name = yaml_path.split("/")[-1].split(".")[0]
data_name = "Prostate"

# Same timepoints as in train.py
timepoints = [
    "T02_Cast_Day7",
    "T03_Cast_Day14",
    "T04_Cast_Day28",
    "T05_Regen_Day1",
    "T06_Regen_Day2",
    "T07_Regen_Day3",
    "T08_Regen_Day7",
    "T09_Regen_Day14",
    "T10_Regen_Day28",
    "T01_Cast_Day1",
]


def run_single_job(args):
    """
    Single training job for one OOD timepoint and seed on a given GPU device.
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
    hparams["OOD"]["filter_dict"]["time"] = ood_cov
    hparams["wandb"]["name"] += f"{ood_cov}_{seed_nb}"
    hparams["train"]["set_seed"] = seed_nb
    hparams["growing_neurons"]["prior_mappers"]["mappers"]["time_mapper"][
        "collapse_name"
    ] = ood_cov

    hparams["evaluations"]["context_transfer_criterion"]["kwargs"][
        "stim_name"
    ] = ood_cov
    hparams["evaluations"]["context_transfer_criterion"]["kwargs"]["subsample"]["gt"][
        "time"
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
    # Use the same seed as in train.py
    seed_nb = 42

    # Distribute ALL OOD covariates across GPUs, with up to 6 concurrent jobs total
    # (3 per GPU on average). Jobs are queued globally: when one finishes, the next
    # ood_cov starts, assigned alternately to GPU 0 and 1.

    jobs = []
    for i, cov in enumerate(timepoints):
        device = i % 2  # even indices -> GPU 0, odd indices -> GPU 1
        jobs.append((cov, seed_nb, device))

    print("Queued jobs (cov, device):", jobs)

    with Pool(processes=4) as pool:
        pool.map(run_single_job, jobs)