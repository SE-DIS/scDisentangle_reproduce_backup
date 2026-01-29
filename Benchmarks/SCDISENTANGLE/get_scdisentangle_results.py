import scanpy as sc
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import anndata as ad
import yaml
import pandas as pd
import os
from icecream import ic

from hygeia.data.load_rnaseq import DatasetLoader
import hygeia.models.mlp_parts as hmlp
import hygeia.utils.config_tools as hconfig
from scdisentangle.train.trainer import Trainer
from scdisentangle.train.CustomIterations import CustomIterations

def infer(
    adata_path, 
    covariate_name, 
    perturbation_name, 
    covariate_names, 
    vars_to_predict,
    control_name,
    stim_name, 
    data_name, 
    yaml_name,
    seeds_list=None,
    ):

    current_dir = os.getcwd()

    adata = sc.read_h5ad(adata_path)
    if covariate_names is None:
        covariate_names = adata.obs[covariate_name].unique().tolist()

    cov_mapper = f'{covariate_name}_mapper'
    pert_mapper_name = f'{perturbation_name}_mapper'

    if seeds_list is None:
        seeds_list = list(range(1, 11))
    for ood_cov in covariate_names:
        for seed_nb in seeds_list:
            
            yaml_path = f"{current_dir}/configs/{yaml_name}.yaml"
            
            with open(yaml_path, 'r') as stream:
                hparams = yaml.safe_load(stream)
            
            hparams['OOD']['filter_dict'][covariate_name] = ood_cov
            hparams['wandb']['wandb_log'] = False
            hparams['train']['set_seed'] = seed_nb
            
            print('ood cov', ood_cov)
            
            hparams['save_experiment']['apply'] = False
            hparams['save_experiment']['save_weights']['apply'] = False
            hparams['save_experiment']['save_best_weights']['apply'] = False
            if not 'sc_cell_ids' in hparams['data']['label_keys']:
                hparams['data']['label_keys'].append('sc_cell_ids')


            dataset = DatasetLoader(
                path=hparams['data']['file_path'],
                label_keys=hparams['data']['label_keys'],
                default_normalization=hparams['data']['default_normalization'],
                min_gene_counts=hparams['data']['min_gene_counts'],
                min_cell_counts=hparams['data']['min_cell_counts'],
                n_highly_variable_genes=hparams['data']['highly_variable'],
                use_counts = hparams['data']['use_counts'],
                subset=hparams['data']['SUBSET'],
            )
        
            # Dataloader cont containing all data samples
            dataloader = dataset.get_dataloader(batch_size=2000)
            
            # Train test val split
            train_dataloader, val_dataloader, test_dataloader = dataset.train_val_ood_key(
                    batch_size=hparams['data']['batch_size'],
                    split_key=f'split_{stim_name}_{ood_cov}'
                    )
        
            trainer = Trainer(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                dataloader=dataloader,
                dataset=dataset,
                device=hparams['hardware']['device'],
                hparams=hparams,
                )
            
            weights_path = f'{current_dir}/weights/{yaml_name}/{seed_nb}/{ood_cov}/r2_mean_criterion_10_all/'
            trainer.load_weights(weights_path)
                
            adata_preds = None
            adata_subset = adata[(adata.obs[perturbation_name] == control_name) & \
                (adata.obs[covariate_name] == ood_cov) & \
                    (adata.obs[f'split_{stim_name}_{ood_cov}'] == 'train')].copy()

            adata_subset
            for dl in ['train']:
                for _var in vars_to_predict:
                    counterfactual_dict = {perturbation_name: _var}

                    hparams['growing_neurons']['prior_mappers']['mappers'][f'{pert_mapper_name}']['collapse_name'] = _var
                                        
                    adata_pred = trainer.predict(
                        adata_subset.copy(), 
                        counterfactual_dict=counterfactual_dict, 
                        bs=256
                    )

                    if adata_preds is None:
                        adata_preds = adata_pred.copy()
                    else:
                        adata_preds = ad.concat([adata_preds, adata_pred])
                    

            _train_adata, _val_adata, _ood_adata = trainer.dataset.train_anndata, trainer.dataset.val_anndata, trainer.dataset.test_anndata
            
            # Compute median
            _sums = _train_adata.X.sum(axis=1, keepdims=True)
            data_median = np.median(_sums)
        
            # Compute means
            train_size, val_size, ood_size = _train_adata.shape[0], _val_adata.shape[0], _ood_adata.shape[0]
            train_mean, val_mean, ood_mean = _train_adata.X.mean(), _val_adata.X.mean(), _ood_adata.X.mean()
            
            # Data loaders
            train_loader_size, val_loader_size, ood_loader_size = len(trainer.train_dataloader), len(trainer.val_dataloader), len(trainer.test_dataloader)

            data_stats = {
                'median': data_median,
                'train_mean': train_mean,
                'val_mean': val_mean,
                'ood_mean': ood_mean,
                'train_size': train_size,
                'val_size': val_size,
                'ood_size': ood_size,
                # number of predicted cells (counterfactual + reconstruction)
                'pred_ood': adata_preds.shape[0],
                'train_loader': train_loader_size,
                'val_loader': val_loader_size,
                'ood_loader': ood_loader_size,
                'X_normalization': 'count',
            }

            adata_preds.uns.update(data_stats)

            directory = f'{current_dir}/predictions/{yaml_name}'

            if not os.path.exists(f'{directory}'):
                os.makedirs(f'{directory}')

            adata_preds.write_h5ad(f'{directory}/{ood_cov}_{seed_nb}.h5ad')