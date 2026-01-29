import sys
import logging
import scanpy as sc
import scgen
import numpy as np
import torch
import random
import wandb
import os
from icecream import ic

import eval_tools

def set_seed(seed):
    ic('Setting seed to', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #os.environ['PYTHONHASHSEED'] = str(seed)
    #sc.settings.seed = seed
    
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

    # Compute train median
    _sums = _train_adata.X.sum(axis=1, keepdims=True)
    data_median = np.median(_sums)

    
    sc.pp.normalize_total(adata, target_sum=data_median)
    sc.pp.log1p(adata)
    
    # Compute means
    train_size, val_size, ood_size = _train_adata.shape[0], _val_adata.shape[0], _ood_adata.shape[0]
    train_mean, val_mean, ood_mean = _train_adata.X.mean(), _val_adata.X.mean(), _ood_adata.X.mean()
    
    # Subset Train
    train_new = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] != 'ood']
    train_new = train_new.copy()

    # Setup data
    scgen.SCGEN.setup_anndata(train_new, batch_key=cond_key, labels_key=cov_key)

    # Init and train SCGEN
    model = scgen.SCGEN(train_new)
    
    model.train(
        max_epochs=100,
        batch_size=32,
        early_stopping=True,
        early_stopping_patience=25
    )

    model.save("model_perturbation_prediction.pt", overwrite=True)
    
    # Create org column for cond
    train_new.obs[f'{cond_key}_org'] = train_new.obs[cond_key].copy()

    # Counterfactual predictions
    adata_preds = None

    for var_to_predict in vars_to_predict:
        pred, delta = model.predict(
            ctrl_key=control_name,
            stim_key=var_to_predict,
            adata_to_predict=train_new.copy() #adata.copy()
            #celltype_to_predict=ood_cov
        )
        pred.obs[f'{cond_key}_pred'] = var_to_predict
    
        preds = sc.AnnData(X=pred.X, obs=pred.obs, var=pred.var)
        if adata_preds is None:
            adata_preds = preds.copy()
        else:
            adata_preds = adata_preds.concatenate(preds)

    gt = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'ood']
    ctrl = adata[(adata.obs[cond_key] == control_name) & (adata.obs[cov_key] == ood_cov)]
    pred = adata_preds[(adata_preds.obs[f'{cond_key}_pred'] == stim_name) & (adata_preds.obs[cov_key] == ood_cov) & (adata_preds.obs[f'{cond_key}_org'] == control_name)]

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
        'X_normalization': 'median',
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