import os
import torch
import numpy as np 
import random
import scanpy as sc
import anndata as ad
from icecream import ic

from scDisInFact import scdisinfact, create_scdisinfact_dataset
from scDisInFact import utils

import gc

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
    ood_cov,
    control_name,
    stim_name,
    categorical_attributes,
    vars_to_predict,
    seed_nb,
    device_nb,
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
    anndatas_save_path = 'predictions/'
    models_save_path = f'models/{ood_cov}_{seed_nb}/'

    # Ensure output directories exist (for every ood_cov / seed_nb)
    os.makedirs(anndatas_save_path, exist_ok=True)
    os.makedirs(models_save_path, exist_ok=True)    
    
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

    # Cells used as sources for counterfactual prediction:
    # control condition, specific covariate, and in the train split.
    input_indices = (
        (meta_cells[cond_key] == control_name)
        & (meta_cells[cov_key] == ood_cov)
        & (meta_cells[f'split_{stim_name}_{ood_cov}'] == 'train')
    )
    
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
        reg_mmd_comm = reg_mmd_comm, 
        reg_mmd_diff = reg_mmd_diff, 
        reg_gl = reg_gl, 
        reg_class = reg_class, 
        reg_kl_comm = reg_kl_comm, 
        reg_kl_diff = reg_kl_diff, 
        seed = seed_nb, 
        device = device,
    )

    # Train
    model.train()

    losses = model.train_model(nepochs = nepochs, recon_loss = "NB")

    # Save model
    torch.save(model.state_dict(), models_save_path + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}_{ood_cov}_{stim_name}_{seed_nb}.pth")

    # Load model
    model.load_state_dict(torch.load(models_save_path + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}_{ood_cov}_{stim_name}_{seed_nb}.pth", map_location = device))

    # Prediction
    _ = model.eval()
    
    adata_preds = None
    for var_to_predict in vars_to_predict:
            meta_input = meta_cells.loc[input_indices, :]
            counts_input = counts[input_indices, :]

            if var_to_predict == control_name:
                predict_conds = None
            else:
                predict_conds = [var_to_predict, ood_cov]
                
            counts_predict = model.predict_counts(
                input_counts = counts_input,
                meta_cells = meta_input,
                condition_keys = categorical_attributes, 
                batch_key = "batch_placeholder",
                predict_conds = predict_conds, 
                predict_batch = None
            )

            adata_pred = sc.AnnData(X = counts_predict, obs=meta_input)
            adata_pred.obs[f'{cond_key}_pred'] = [var_to_predict] * adata_pred.shape[0]
            if adata_preds is None:
                adata_preds = adata_pred.copy()
            else:
                adata_preds = ad.concat([adata_preds, adata_pred])

    adata_preds.obs[cond_key] = adata_preds.obs[f'{cond_key}_pred'].copy()
        
    data_stats = {
            'median': data_median,
            'train_mean': train_mean,
            'val_mean': val_mean,
            'ood_mean': ood_mean,
            'train_size': train_size,
            'val_size': val_size,
            'ood_size': ood_size,
        
            'pred_ood': adata_preds.shape[0],
            # number of source control cells used for prediction
            'adata_subset_shape': int(input_indices.sum()),
        
            'X_normalization': 'count',
        }
        
    adata_preds.uns.update(data_stats)
    
    adata_preds.write_h5ad(f'{anndatas_save_path}{ood_cov}_{seed_nb}.h5ad')

    del _train_adata, _val_adata, _ood_adata, adata_preds, adata
    gc.collect()