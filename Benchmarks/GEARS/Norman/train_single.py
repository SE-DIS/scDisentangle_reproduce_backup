import os
import json
import train_gears_norman

custom_name = 'single_only'
data_name = 'Norman'
adata_path = '../../../Datasets/preprocessed_datasets/norman.h5ad'

conds = [
    'CBL+CNN1', 'DUSP9+ETS2', 'DUSP9+MAPK1', 
    'ETS2+MAPK1', 'ETS2+CEBPE', 'CNN1+MAPK1', 
    'CEBPB+CEBPA', 'CBL+PTPN12', 'AHR+FEV', 
]

with open(f'../../../Datasets/preprocessed_datasets/norman_splits_2.json') as f:
    _split_dict = json.load(f)

run_indices = list(_split_dict.keys())
seed_nb = 42

for split_index in run_indices:
    split_dict = _split_dict[split_index]

    # Create path ?
    if os.path.isfile(f'predictions/{custom_name}/{split_index}.json'):
        print(split_index, seed_nb, 'already processed, skipping')
        continue

    train_gears_norman.train(
        adata_path=adata_path,
        ood_index=split_index,
        seed_nb=seed_nb,
        split_dict=split_dict,
        vars_to_predict=[conds[int(split_index)]],
        custom_name=custom_name
        )