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

import gc

def infer(
    adata_path, 
    vars_to_predict,
    yaml_name,
    seed_nb,
    ood_index,
    custom_name,
    split_dict
    ):

    current_dir = os.getcwd()

    # Read adata
    adata = sc.read_h5ad(adata_path)
    
    # Perturbation mappers
    pert1_mapper = 'perturbation1_mapper'
    pert2_mapper = 'perturbation2_mapper'
    
    # Yaml path
    yaml_path = f"{current_dir}/configs/{yaml_name}.yaml"
    
    # Read params
    with open(yaml_path, 'r') as stream:
        hparams = yaml.safe_load(stream)
    
    # Set seed
    hparams['train']['set_seed'] = seed_nb

    # Turn off logs and save
    hparams['wandb']['wandb_log'] = False 
    hparams['save_experiment']['apply'] = False
    hparams['save_experiment']['save_weights']['apply'] = False
    hparams['save_experiment']['save_best_weights']['apply'] = False
    
    # Get dataset and dataloaders  
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

    train_mask = dataset.data.obs['condition'].isin(split_dict['train'])
    val_mask = dataset.data.obs['condition'].isin(split_dict['val'])
    ood_mask = dataset.data.obs['condition'].isin(split_dict['ood'])

    dataset.data.obs['split_ood'] = None
    dataset.data.obs['split_ood'][train_mask] = 'train'
    dataset.data.obs['split_ood'][val_mask] = 'val'
    dataset.data.obs['split_ood'][ood_mask] = 'ood'
    
    # Dataloader cont containing all data samples
    dataloader = dataset.get_dataloader(batch_size=2000)
            
    train_dataloader, val_dataloader, test_dataloader, ctrl_dataloader = dataset.train_val_ood_key(
        batch_size=hparams['data']['batch_size'],
        split_key='split_ood',
        add_ctrl_dataloader=True,
        ctrl_name='ctrl',
        cond_key='condition'
    )
    
    # Get trainer
    trainer = Trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        ctrl_dataloader=ctrl_dataloader,
        dataloader=dataloader,
        dataset=dataset,
        device=hparams['hardware']['device'],
        hparams=hparams,
        )
    
    # Load weights
    weights_path = f'{current_dir}/weights/{custom_name}/{seed_nb}/{ood_index}/r2_mean_criterion_10_all/'
    trainer.load_weights(weights_path)
        
    # Infer
    adata_subset = adata[adata.obs['cond_harm'] == 'ctrl']

    # predict control
    adata_pred_ctrl = trainer.predict(
            adata_subset.copy(), 
            counterfactual_dict={
                'perturbation1':'ctrl', 
                'perturbation2':'NOPERT'
            }, 
            bs=256
        )
    
    adata_pred_ctrl.obs['cond_harm'] = ['ctrl'] * adata_pred_ctrl.shape[0]
    adata_pred_ctrl.obs['cond_harm_pred'] = ['ctrl'] * adata_pred_ctrl.shape[0]

    # predict ood
    for _var in vars_to_predict:
        pert1, pert2 = _var.split('+')

        counterfactual_dict = {
            'perturbation1': pert1,
            'perturbation2': pert2,
            }
                                    
        adata_pred = trainer.predict(
            adata_subset.copy(), 
            counterfactual_dict=counterfactual_dict, 
            bs=256
        )

        adata_pred.obs['cond_harm'] = [_var] * adata_pred.shape[0]
        adata_pred.obs['cond_harm_pred'] = [_var] * adata_pred.shape[0]

        counterfactual_dict_pert1 = {
            'perturbation1': pert1,
            'perturbation2': 'NOPERT',
            }
                                    
        adata_pred_pert1 = trainer.predict(
            adata_subset.copy(), 
            counterfactual_dict=counterfactual_dict_pert1,
            bs=256
        )

        adata_pred_pert1.obs['cond_harm'] = [pert1] * adata_pred_pert1.shape[0]
        adata_pred_pert1.obs['cond_harm_pred'] = [pert1] * adata_pred_pert1.shape[0]

        counterfactual_dict_pert2 = {
            'perturbation1': pert2,
            'perturbation2': 'NOPERT',
            }
                                    
        adata_pred_pert2 = trainer.predict(
            adata_subset.copy(), 
            counterfactual_dict=counterfactual_dict_pert2,
            bs=256
        )

        adata_pred_pert2.obs['cond_harm'] = [pert2] * adata_pred_pert2.shape[0]
        adata_pred_pert2.obs['cond_harm_pred'] = [pert2] * adata_pred_pert2.shape[0]
        
        data_stats = {
            'X_normalization': 'count',
        }

        adata_pred = ad.concat([adata_pred, adata_pred_ctrl, adata_pred_pert1, adata_pred_pert2])
        adata_pred.uns.update(data_stats)

        directory = f'{current_dir}/predictions/{custom_name}'

        if not os.path.exists(f'{directory}'):
            os.makedirs(f'{directory}')

        adata_pred.write_h5ad(f'{directory}/{_var}.h5ad')

    # Free memory
    del adata_pred, adata_subset, adata, trainer.dataset.data, trainer.dataset.train_anndata
    gc.collect()