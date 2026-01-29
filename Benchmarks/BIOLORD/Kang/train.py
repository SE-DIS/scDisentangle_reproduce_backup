import sys
import os
sys.path.append('../')

import train_biolord
from tqdm import tqdm

cov_key = 'cell_type'
cond_key = 'condition'
control_name = 'control'
stim_name = 'stimulated'
vars_to_predict = ['stimulated', 'control']
categorical_attributes = ['condition', 'cell_type']

adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/kang.h5ad'

cell_types = ['B', 'T', 'NK', 'CD4 T', 'CD8 T', 'DC', 'CD14 Mono', 'CD16 Mono']

for ood_cov in cell_types:

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