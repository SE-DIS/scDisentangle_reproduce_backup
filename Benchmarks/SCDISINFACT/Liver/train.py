import sys
import os
sys.path.append('../')

import train_scdisinfact
from tqdm import tqdm

cov_key = 'zone'
cond_key = 'status_control'
control_name = 'Control'
stim_name = 'Infected'
vars_to_predict = ['Infected', 'Uninfected', 'Control']
categorical_attributes = ['status_control', 'zone'] # Should be in this order: cond, cov

adata_path ='/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/liver.h5ad'
device_nb = 1

zones = ['Periportal', 'Pericentral']

for ood_cov in zones:

    for seed_nb in tqdm(range(1, 11)):
        
        if os.path.isfile(f'predictions/{ood_cov}_{seed_nb}.h5ad'):
            print(ood_cov, seed_nb, 'already processed, skipping')
            continue
            
        train_scdisinfact.train(
            adata_path = adata_path,
            cov_key = cov_key,
            cond_key = cond_key,
            control_name = control_name,
            stim_name = stim_name,
            vars_to_predict = vars_to_predict,
            ood_cov = ood_cov,
            categorical_attributes = categorical_attributes,
            seed_nb = seed_nb,
            device_nb=device_nb
        )