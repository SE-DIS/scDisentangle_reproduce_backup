import sys, os
import torch
import numpy as np 
import random
import pandas as pd
import scanpy as sc
import wandb
from icecream import ic

from scDisInFact import scdisinfact, create_scdisinfact_dataset
from scDisInFact import utils
import eval_tools

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
    adata_path,
    data_name,
    cov_key,
    cond_key,
    ood_cov,
    control_name,
    stim_name,
    categorical_attributes,
    vars_to_predict,
    seed_nb,
    device_nb,
    wandb_infos
    ):
    # Set Device
    device =  torch.device(f"cuda:{device_nb}" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    set_seed(seed_nb)

    # Read adata
    adata = sc.read_h5ad(adata_path)

    try:
        adata.X = adata.X.toarray()
    except:
        print('adata matrix is already in array format')

    # Create save directories
    anndatas_save_path = f'{save_path}/pred_adata/'
    additional_save_path = f'{save_path}/additional/'

    if not os.path.exists(anndatas_save_path):
        os.makedirs(anndatas_save_path)
        os.makedirs(additional_save_path)        

    result_dir = 'weights/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    # Create batch placeholder
    adata.obs['batch_placeholder'] = [0] * adata.shape[0]

    # Org column
    adata.obs[f'{cond_key}_org'] = adata.obs[cond_key].copy() 
    
    # Capture counts to numpy and metadata to df
    counts = adata.X
    meta_cells = adata.obs

    # Convert columns to str
    for _col in categorical_attributes:
        meta_cells[_col] = meta_cells[_col].astype(str)
    
    # Train and test ood
    test_idx = (meta_cells[f'split_{stim_name}_{ood_cov}'] == 'ood')
    train_idx = ~test_idx

    # Compute data stats (train / val / ood)
    _train_adata = counts[train_idx, :]
    _val_adata = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'val'].copy()
    _ood_adata = counts[test_idx, :]

    # Compute median
    _sums = _train_adata.sum(axis=1, keepdims=True)
    data_median = np.median(_sums)

    # Compute means
    train_size, val_size, ood_size = _train_adata.shape[0], _val_adata.shape[0], _ood_adata.shape[0]
    train_mean, val_mean, ood_mean = _train_adata.mean(), _val_adata.X.mean(), _ood_adata.mean()
    
    # Create scDisInFact dataset
    data_dict = create_scdisinfact_dataset(
        counts[train_idx,:], 
        meta_cells.loc[train_idx,:], 
        condition_key = categorical_attributes, 
        batch_key = "batch_placeholder"
    )

    # default setting of hyper-parameters
    reg_mmd_comm = 1e-4
    reg_mmd_diff = 1e-4
    reg_kl_comm = 1e-5
    reg_kl_diff = 1e-2
    reg_class = 1
    reg_gl = 1
        
    Ks = [8] + [2] * len(categorical_attributes) # Dimension of latent factors (shared bio factor, unshared1, unshared2..)
    # These values follow paper recommendation and demo tutorial
    
    batch_size = 64
    nepochs = 100
    interval = 10
    lr = 5e-4

    lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]

    # scDisInFact Model
    model = scdisinfact(
        data_dict = data_dict, 
        Ks = Ks, 
        batch_size = batch_size, 
        interval = interval, 
        lr = lr, 
        reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
        reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = seed_nb, device = device,
        
    )

    # Train
    model.train()

    losses = model.train_model(nepochs = nepochs, recon_loss = "NB")

    # Save model
    torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}_{ood_cov}_{stim_name}_{seed_nb}.pth")

    # Load model
    model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}_{ood_cov}_{stim_name}_{seed_nb}.pth", map_location = device))

    # Prediction
    _ = model.eval()
    
    adata_preds = None
    for var_to_predict in vars_to_predict:
        for _cov in np.unique(adata.obs[cov_key]):
            input_idx = (meta_cells[cov_key] == _cov)
            meta_input = meta_cells.loc[input_idx, :]
            counts_input = counts[input_idx, :]
            
            counts_predict = model.predict_counts(
                input_counts = counts_input, 
                meta_cells = meta_input,
                condition_keys = categorical_attributes, 
                batch_key = "batch_placeholder",
                predict_conds = [var_to_predict, _cov], 
                predict_batch = 0
            )

            adata_pred = sc.AnnData(X = counts_predict, obs=meta_input)
            adata_pred.obs[f'{cond_key}_pred'] = [var_to_predict] * adata_pred.shape[0]
            if adata_preds is None:
                adata_preds = adata_pred.copy()
            else:
                adata_preds = adata_preds.concatenate(adata_pred)
        
        # pred and gt ood, eval them (preliminary)
    pred = adata_preds[(adata_preds.obs[f'{cond_key}_pred'] == stim_name) & (adata_preds.obs[cov_key] == ood_cov) & (adata_preds.obs[f'{cond_key}_org'] == control_name)].copy()
    gt = adata[(adata.obs[cov_key] == ood_cov) & (adata.obs[f'{cond_key}'] == stim_name)].copy()
    ctrl = adata[(adata.obs[cov_key] == ood_cov) & (adata.obs[f'{cond_key}'] == control_name)].copy()
    
    pred_median = pred.copy()
    gt_median = gt.copy()
        
    sc.pp.normalize_total(pred)
    sc.pp.log1p(pred)
    
    sc.pp.normalize_total(gt)
    sc.pp.log1p(gt)
        
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
        
    data_stats = {
            'median': data_median,
            'train_mean': train_mean,
            'val_mean': val_mean,
            'ood_mean': ood_mean,
            'train_size': train_size,
            'val_size': val_size,
            'ood_size': ood_size,
            'pred_ood': pred.shape[0],
            'X_normalization': 'count',
        }
        
    adata_preds.uns.update(data_stats)
    adata_preds.write_h5ad(f'{anndatas_save_path}{ood_cov}_{seed_nb}.h5ad')