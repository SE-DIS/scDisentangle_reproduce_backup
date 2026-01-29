import sys
import json

from get_results_norman import infer

if __name__ == '__main__':
    adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/norman.h5ad'

    with open(f'../../../Datasets/preprocessed_datasets/norman_splits_1.json') as f:
        _split_dict = json.load(f)

    seed_nb = 42
    run_indices = list(_split_dict.keys())
    for split_index in run_indices:
        split_dict = _split_dict[split_index]
        vars_to_predict = split_dict['ood']

        infer(
            adata_path=adata_path,  
            vars_to_predict=vars_to_predict, 
            yaml_name='norman',
            seed_nb=seed_nb,
            ood_index=split_index,
            custom_name='combinatorially_seen',
            split_dict=split_dict,
            )