import sys
import os
sys.path.append('../')

import train_scdisinfact
from tqdm import tqdm

cov_key = 'cell_type'
cond_key = 'specie'
control_name = 'muris'
stim_name = 'sapiens'
vars_to_predict = ['sapiens', 'muris']
categorical_attributes = ['specie', 'cell_type'] # Should be in this order: cond, cov

adata_path ='/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/tabula.h5ad'
device_nb = 1

cell_types = ['pulmonary alveolar type 2 cell']

for ood_cov in cell_types:

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