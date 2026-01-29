import sys
sys.path.append('../')

from get_scdisentangle_results import infer

if __name__ == '__main__':
    adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/liver.h5ad'
    covariate_name = 'zone'
    perturbation_name = 'infected'
    covariate_names = ['Periportal', 'Pericentral']
    vars_to_predict = ['TRUE', 'FALSE']
    control_name = 'FALSE'
    stim_name = 'TRUE'
    data_name = 'Liver'
    yaml_name = 'liver_infected'

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