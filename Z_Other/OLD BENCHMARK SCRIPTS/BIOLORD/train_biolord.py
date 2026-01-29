import pandas as pd
import numpy as np
import scanpy as sc
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import biolord
import wandb
import os
import sys
from icecream import ic

import eval_tools

def total_to_median_norm(_adata, data_median):
        _adata.X = np.expm1(_adata.X)
        _adata.X = _adata.X / 1e4
        _adata.X = _adata.X * data_median
    
        sc.pp.log1p(_adata)
        return _adata
    
def set_seed(seed):
    ic('Setting seed to', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train(
    save_path,
    data_name,
    adata_path,
    cov_key,
    cond_key,
    control_name,
    stim_name,
    vars_to_predict,
    ood_cov,
    categorical_attributes,
    seed_nb,
    wandb_infos=None,
    ):

    # Read adata
    adata = sc.read_h5ad(adata_path)

    # Set seeed
    set_seed(seed_nb)

    # Create save directories
    anndatas_save_path = f'{save_path}/pred_adata/'
    additional_save_path = f'{save_path}/additional/'

    if not os.path.exists(anndatas_save_path):
        os.makedirs(anndatas_save_path)
        os.makedirs(additional_save_path)        

    # Log train / val / ood statistics
    _train_adata = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'train'].copy()
    _val_adata = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'val'].copy()
    _ood_adata = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'ood'].copy()

    # Replace val by test
    adata.obs[f'split_{stim_name}_{ood_cov}'] = [x.replace('val', 'test') for x in adata.obs[f'split_{stim_name}_{ood_cov}']]
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Compute train median
    _sums = _train_adata.X.sum(axis=1, keepdims=True)
    data_median = np.median(_sums)

    # Compute means
    train_size, val_size, ood_size = _train_adata.shape[0], _val_adata.shape[0], _ood_adata.shape[0]
    train_mean, val_mean, ood_mean = _train_adata.X.mean(), _val_adata.X.mean(), _ood_adata.X.mean()

    # Setup Biolord data
    biolord.Biolord.setup_anndata(
        adata,
        ordered_attributes_keys=None,
        categorical_attributes_keys= categorical_attributes,
    )

    # Params
    module_params = {
        "decoder_width": 1024,
        "decoder_depth": 4,
        "attribute_nn_width": 512,
        "attribute_nn_depth": 2,
        "n_latent_attribute_categorical": 4,
        "gene_likelihood": "normal",
        "reconstruction_penalty": 1e2,
        "unknown_attribute_penalty": 1e1,
        "unknown_attribute_noise_param": 1e-1,
        "attribute_dropout_rate": 0.1,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "seed": seed_nb,
    }

    # Init Biolord model
    model = biolord.Biolord(
        adata=adata,
        n_latent=32,
        model_name=data_name,
        module_params=module_params,
        train_classifiers=False,
        split_key=f'split_{stim_name}_{ood_cov}',
    )

    # Trainer params
    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": 1e-4,
        "latent_wd": 1e-4,
        "decoder_lr": 1e-4,
        "decoder_wd": 1e-4,
        "attribute_nn_lr": 1e-2,
        "attribute_nn_wd": 4e-8,
        "step_size_lr": 45,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
    }

    # Train
    model.train(
        max_epochs=500,
        batch_size=512,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=20,
        check_val_every_n_epoch=10,
        num_workers=1,
        enable_checkpointing=False,
    )

    # Create org cond_key
    adata.obs[f'{cond_key}_org'] = adata.obs[cond_key].copy() 
    adata_source = adata.copy()

    # Predict cond_key with all possible attributes
    adata_preds = model.compute_prediction_adata(
        adata, 
        adata_source, 
        target_attributes=[cond_key], 
        add_attributes=[cov_key, f'{cond_key}_org', f'split_{stim_name}_{ood_cov}', 'sc_cell_ids']
    )

    # Create pred attribute column
    adata_preds.obs[f'{cond_key}_pred'] = adata_preds.obs[cond_key].copy()

    # Subset it to selected attributes e.g. Control and Infected, exclude Uninfected.
    adata_preds = adata_preds[adata_preds.obs[f'{cond_key}_pred'].isin(vars_to_predict)].copy()

    adata = total_to_median_norm(adata, data_median)
    adata_preds = total_to_median_norm(adata_preds, data_median)
    # Subset prediction and ground truth (ood stim)
    pred = adata_preds[(adata_preds.obs[f'{cond_key}_pred'] == stim_name) & (adata_preds.obs[cov_key] == ood_cov) & (adata_preds.obs[f'{cond_key}_org'] == control_name)]
    gt = adata[(adata.obs[cov_key] == ood_cov) & (adata.obs[f'{cond_key}_org'] == stim_name)]
    ctrl = adata[(adata.obs[cov_key] == ood_cov) & (adata.obs[f'{cond_key}_org'] == control_name)].copy()

    # Log data stats / infos
    data_stats = {
        'median': data_median,
        'train_mean': train_mean,
        'val_mean': val_mean,
        'ood_mean': ood_mean,
        'train_size': train_size,
        'val_size': val_size,
        'ood_size': ood_size,
        'pred_ood': pred.shape[0],
        'X_normalization': 'target_1e4',
    }
    adata_preds.uns.update(data_stats)

    # Compute preliminary metrics and log in wandb (norm_target)
    degs = adata.uns[f'rank_genes_groups_{cond_key}'][stim_name][ood_cov]
    degs_indices = [adata.var_names.get_loc(x) for x in degs]
    pred = pred.X
    gt = gt.X
    ctrl = ctrl.X
    
    metrics = eval_tools.get_metrics(
        degs_indices=degs_indices,
        pred=pred,
        gt=gt,
        ctrl=ctrl
    )
    
    if wandb_infos is not None:
        wandb.init(
            name=wandb_infos['name'], 
            group=wandb_infos['group'],
            project=wandb_infos['project']
            )
        wandb.log(metrics)
        wandb.finish()

    # Write adata_preds
    adata_preds.write_h5ad(f'{anndatas_save_path}{ood_cov}_{seed_nb}.h5ad')