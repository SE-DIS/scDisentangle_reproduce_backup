import sys
import os
sys.path.append('../')
sys.path.append('../../../')

import train_scdisinfact
from tqdm import tqdm

zones = ['B', 'T', 'NK', 'CD4 T', 'CD8 T', 'DC', 'CD14 Mono', 'CD16 Mono']

data_name = 'Kang'
custom_name = 'condition'

cov_key = 'cell_type'
cond_key = 'condition'
control_name = 'control'
stim_name = 'stimulated'
vars_to_predict = ['stimulated', 'control']
categorical_attributes = ['condition', 'cell_type'] # Should be in this order: cond, cov
adata_path ='/data/Experiments/Benchmark/scdisentangle/Datasets/preprocessed_datasets/kang.h5ad'
device_nb = 1

for ood_cov in zones:

    for seed_nb in tqdm(range(1, 11)):
        
        wandb_infos = {
            'name': f'{ood_cov}_{seed_nb}',
            'group': f'SCDISINFACT_{data_name}_{custom_name}',
            'project': 'SCBENCHMARKS'
        }

        save_path = f'../../Datasets/{data_name}/{custom_name}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if os.path.isfile(f'{save_path}/pred_adata/{ood_cov}_{seed_nb}.h5ad'):
            print(ood_cov, seed_nb, 'already processed, skipping')
            continue
            
        train_scdisinfact.train(
            save_path = save_path,
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
            wandb_infos = wandb_infos,
            device_nb=device_nb
        )