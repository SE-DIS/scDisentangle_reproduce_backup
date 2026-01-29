import sys
import os
sys.path.append('../')

import train_biolord
from tqdm import tqdm

cov_key = 'zone'
cond_key = 'infected'
control_name = 'FALSE'
stim_name = 'TRUE'
vars_to_predict = ['TRUE', 'FALSE']
categorical_attributes = ['infected', 'zone']

adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/liver.h5ad'

zones = ['Periportal', 'Pericentral']

for ood_cov in zones:

    for seed_nb in tqdm(range(1, 11)):
        
        if os.path.isfile(f'predictions/{ood_cov}_{seed_nb}.h5ad'):
            print(ood_cov, seed_nb, 'already processed, skipping')
            continue
        
        train_biolord.train(
            adata_path = adata_path,
            cov_key = cov_key,
            cond_key = cond_key,
            control_name = control_name,
            stim_name = stim_name,
            vars_to_predict = vars_to_predict,
            ood_cov = ood_cov,
            categorical_attributes = categorical_attributes,
            seed_nb = seed_nb,
        )