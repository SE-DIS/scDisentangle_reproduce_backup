import sys
import os
from tqdm import tqdm

sys.path.append('../')

import train_cpa

data_name = 'Myocarditis_org'


cov_key = 'donor'
cond_key = 'tissue'
control_name = 'Blood'
stim_name = 'Heart'
vars_to_predict = ['Heart', 'Blood']
categorical_attributes = ['donor']

adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/myocarditis_org.h5ad'
zones = ['SIC_264', 'SIC_171', 'SIC_258', 'SIC_153', 'SIC_48', 'SIC_232', 'SIC_177'] #['SIC_264', 'SIC_171', 'SIC_258', 'SIC_177', 'SIC_48', 'SIC_232']

for ood_cov in zones:

    for seed_nb in tqdm(range(1, 11)):
        
        if os.path.isfile(f'predictions/{ood_cov}_{seed_nb}.h5ad'):
            print(ood_cov, seed_nb, 'already processed, skipping')
            continue
            
        train_cpa.train(
            data_name = data_name,
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