import sys
#sys.path.append('../')

from get_scdisentangle_results import infer

if __name__ == '__main__':
    adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/icis.h5ad'
    covariate_name = 'patient_id'
    perturbation_name = 'condition'
    covariate_names = ['P1','P3','P20','P9','P11','P10','P4','P5']
    vars_to_predict = ['On treatment', 'Baseline']
    control_name = 'Baseline'
    stim_name = 'On treatment'
    data_name = 'ici'
    yaml_name = 'ici'

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