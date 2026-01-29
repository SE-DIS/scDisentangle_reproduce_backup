import sys
sys.path.append('../')

from get_scdisentangle_results import infer

if __name__ == '__main__':
    adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/tabula.h5ad'
    covariate_name = 'cell_type'
    perturbation_name = 'specie'
    covariate_names = ['pulmonary alveolar type 2 cell']
    vars_to_predict = ['sapiens', 'muris']
    control_name = 'muris'
    stim_name = 'sapiens'
    data_name = 'Tabula'
    yaml_name = 'tabula'

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