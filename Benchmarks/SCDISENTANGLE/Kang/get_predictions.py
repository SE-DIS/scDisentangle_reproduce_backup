import sys
sys.path.append('../')

from get_scdisentangle_results import infer

if __name__ == '__main__':
    adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/kang.h5ad'
    covariate_name = 'cell_type'
    perturbation_name = 'condition'
    covariate_names = ['B', 'T', 'NK', 'CD4 T', 'CD8 T', 'CD14 Mono', 'CD16 Mono', 'DC']
    vars_to_predict = ['stimulated', 'control']
    control_name = 'control'
    stim_name = 'stimulated'
    data_name = 'Kang'
    yaml_name = 'kang'

    infer(
        adata_path=adata_path, 
        covariate_name=covariate_name, 
        perturbation_name=perturbation_name, 
        covariate_names=covariate_names, 
        vars_to_predict=vars_to_predict, 
        control_name=control_name,
        stim_name=stim_name, 
        data_name=data_name, 
        yaml_name=yaml_name
        )