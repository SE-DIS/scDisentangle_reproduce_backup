import scanpy as sc
import anndata as ad
import scgen
import numpy as np
import torch
import random
import os
from icecream import ic
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
    control_name,
    stim_name,
    vars_to_predict,
    ood_cov,
    seed_nb,
    ):
    
    # Read adata
    adata = sc.read_h5ad(adata_path)

    try:
        adata.X = adata.X.toarray()
    except:
        print('Data is already array')
    
    # Set seeed
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

    # Compute train median
    _sums = _train_adata.X.sum(axis=1, keepdims=True)
    data_median = np.median(_sums)

    # Normalize by train median
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
    model.save(f'{models_save_path}model.pt', overwrite=True)
    
    # Train & val indices
    split_key = f'split_{stim_name}_{ood_cov}'
    train_idx = np.where(train_new.obs[split_key] == 'train')[0]
    val_idx   = np.where(train_new.obs[split_key] == 'val')[0]
    test_idx  = np.array([], dtype=int)
    ic(len(train_idx), len(val_idx), len(test_idx))

    model.train(
        max_epochs=100,
        batch_size=32,
        early_stopping=True,
        early_stopping_patience=25,
        
    )

    # Save model
    model.save(f'{models_save_path}model.pt', overwrite=True)
    
    # Create org column for cond
    train_new.obs[f'{cond_key}_org'] = train_new.obs[cond_key].copy()

    # Counterfactual predictions
    adata_preds = None

    adata_subset = adata[(adata.obs[cond_key] == control_name) & (adata.obs[cov_key] == ood_cov) & (adata.obs[f'split_{stim_name}_{ood_cov}'] == 'train')].copy()
    
    for var_to_predict in vars_to_predict:
        # Predict expression under target condition `var_to_predict`
        pred, delta = model.predict(
            ctrl_key=control_name,
            stim_key=var_to_predict,
            adata_to_predict=adata_subset.copy(),
        )

        pred.obs[f'{cond_key}_org'] = adata_subset.obs[cond_key].values
        pred.obs[cond_key] = var_to_predict
        pred.obs[f'{cond_key}_pred'] = var_to_predict

        adata_pred = sc.AnnData(X=pred.X, obs=pred.obs, var=pred.var)
        if adata_preds is None:
            adata_preds = adata_pred.copy()
        else:
            adata_preds = ad.concat([adata_preds, adata_pred])

        # Assert it's a reconstruction
        # if var_to_predict == control_name:
        #     assert np.allclose(delta, 0, atol=1e-6)

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

    # Write adata_preds
    adata_preds.write_h5ad(f'{anndatas_save_path}{ood_cov}_{seed_nb}.h5ad')
    
    del _train_adata, _val_adata, _ood_adata, adata_preds, adata
    gc.collect()