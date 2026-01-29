import numpy as np
import scanpy as sc
import anndata as ad
import torch
import random

import biolord
import os
from icecream import ic
import gc

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
    adata_path,
    cov_key,
    cond_key,
    control_name,
    stim_name,
    vars_to_predict,
    ood_cov,
    categorical_attributes,
    seed_nb,
    ):

    # Read adata
    adata = sc.read_h5ad(adata_path)
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()

    # Set seed
    set_seed(seed_nb)

    # Create save directories
    anndatas_save_path = 'predictions/'
    models_save_path = f'models/{ood_cov}_{seed_nb}/'

    # Ensure output directories exist (for every ood_cov / seed_nb)
    os.makedirs(anndatas_save_path, exist_ok=True)
    os.makedirs(models_save_path, exist_ok=True)    

    # Log train / val / ood statistics
    _train_adata = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'train'].copy()
    _val_adata = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'val'].copy()
    _ood_adata = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'ood'].copy()

    # Compute means
    train_size, val_size, ood_size = _train_adata.shape[0], _val_adata.shape[0], _ood_adata.shape[0]
    train_mean, val_mean, ood_mean = _train_adata.X.mean(), _val_adata.X.mean(), _ood_adata.X.mean()
    
    # Compute train median
    _sums = _train_adata.X.sum(axis=1, keepdims=True)
    data_median = np.median(_sums)

    # Replace val by test
    adata.obs[f'split_{stim_name}_{ood_cov}'] = [x.replace('val', 'test') for x in adata.obs[f'split_{stim_name}_{ood_cov}']]
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
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
        model_name=f'{ood_cov}_{seed_nb}',
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
    
    adata_subset = adata[(adata.obs[cond_key] == control_name) & (adata.obs[cov_key] == ood_cov) & (adata.obs[f'split_{stim_name}_{ood_cov}'] == 'train')].copy()

    adata_source = adata_subset.copy()
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
    adata_preds.obs[cond_key] = adata_preds.obs[f'{cond_key}_pred'].copy()

    adata_preds = total_to_median_norm(adata_preds, data_median)
    
    # Log data stats / infos
    data_stats = {
        'median': data_median,
        'train_mean': train_mean,
        'val_mean': val_mean,
        'ood_mean': ood_mean,
        'train_size': train_size,
        'val_size': val_size,
        'ood_size': ood_size,
        
        'pred_ood': adata_preds.shape[0],
        'adata_subset_shape': adata_subset.shape[0],
        
        'X_normalization': 'median',
    }
    adata_preds.uns.update(data_stats)

    # Replace 'test' by 'val' in the split column on the prediction AnnData
    adata_preds.obs[f'split_{stim_name}_{ood_cov}'] = [
        x.replace('test', 'val') for x in adata_preds.obs[f'split_{stim_name}_{ood_cov}']
    ]
    
    # Write adata_preds
    adata_preds.write_h5ad(f'{anndatas_save_path}{ood_cov}_{seed_nb}.h5ad')

    del _train_adata, _val_adata, _ood_adata, adata_preds, adata
    gc.collect()