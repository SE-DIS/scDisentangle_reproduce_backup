import sys
sys.path.append('../')

from get_scdisentangle_results import infer

if __name__ == '__main__':
    adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/myocarditis.h5ad'
    covariate_name = 'donor'
    perturbation_name = 'tissue'
    covariate_names = ['SIC_264', 'SIC_171', 'SIC_258', 'SIC_177', 'SIC_48', 'SIC_232'] #['SIC_264', 'SIC_171', 'SIC_258', 'SIC_153', 'SIC_48', 'SIC_232', 'SIC_177']
    vars_to_predict = ['Heart', 'Blood']
    control_name = 'Blood'
    stim_name = 'Heart'
    data_name = 'Myocarditis'
    yaml_name = 'myocarditis'

    infer(
        adata_path=adata_path, 
        covariate_name=covariate_name, 
        perturbation_name=perturbation_name, 
        covariate_names=covariate_names, 
        vars_to_predict=vars_to_predict, 
        control_name=control_name,
        stim_name=stim_name, 
        data_name=data_name, 
        yaml_name=yaml_name,
        #seeds_list=[42]
        )