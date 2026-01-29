import sys
import json

from get_results_norman import infer

if __name__ == '__main__':
    adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/norman.h5ad'
    conds = [
        'CBL+CNN1', 'DUSP9+ETS2', 'DUSP9+MAPK1', 
        'ETS2+MAPK1', 'ETS2+CEBPE', 'CNN1+MAPK1', 
        'CEBPB+CEBPA', 'CBL+PTPN12', 'AHR+FEV', 
    ]
    
    with open(f'../../../Datasets/preprocessed_datasets/norman_splits_2.json') as f:
        _split_dict = json.load(f)

    seed_nb = 42
    run_indices = list(_split_dict.keys())
    for i, split_index in enumerate(run_indices):
        assert i == int(split_index)
        split_dict = _split_dict[split_index]
        vars_to_predict = [conds[i]]
        
        infer(
            adata_path=adata_path,  
            vars_to_predict=vars_to_predict, 
            yaml_name='norman',
            seed_nb=seed_nb,
            ood_index=split_index,
            custom_name='single_only',
            split_dict=split_dict,
            )