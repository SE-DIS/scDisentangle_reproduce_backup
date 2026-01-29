import sys
sys.path.append('../')

from get_scdisentangle_results import infer

if __name__ == '__main__':
    adata_path = '/data/Experiments/Benchmark/SCDISENTANGLE_REPRODUCE/Datasets/preprocessed_datasets/myocarditis_org_pre_steroid.h5ad'
    covariate_name = 'donor'
    perturbation_name = 'tissue'
    covariate_names = ["SIC_258","SIC_199","SIC_264","SIC_48","SIC_197","SIC_164","SIC_217","SIC_153","SIC_171"]
    
    vars_to_predict = ['Heart', 'Blood']
    control_name = 'Blood'
    stim_name = 'Heart'
    data_name = 'Myocarditis_org'
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