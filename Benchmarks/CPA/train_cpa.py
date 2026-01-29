import numpy as np
import scanpy as sc
import torch
import random
import cpa
import os
from icecream import ic
import anndata as ad
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
    ):
    print('Starting training for', ood_cov, 'with seed', seed_nb)
    # Read adata
    adata = sc.read_h5ad(adata_path)

    try:
        adata.X = adata.X.toarray()
    except:
        print('Data is already array')

    # Set seed
    set_seed(seed_nb)

    # Create save directories
    anndatas_save_path = 'predictions/'
    models_save_path = f'models/{ood_cov}_{seed_nb}/'

    os.makedirs(anndatas_save_path, exist_ok=True)
    os.makedirs(models_save_path, exist_ok=True)        

    # Compute data stats (train / val / ood)
    _train_adata = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'train'].copy()
    _val_adata = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'val'].copy()
    _ood_adata = adata[adata.obs[f'split_{stim_name}_{ood_cov}'] == 'ood'].copy()

    # Compute median
    _sums = _train_adata.X.sum(axis=1, keepdims=True)
    data_median = np.median(_sums)

    # Compute means
    train_size, val_size, ood_size = _train_adata.shape[0], _val_adata.shape[0], _ood_adata.shape[0]
    train_mean, val_mean, ood_mean = _train_adata.X.mean(), _val_adata.X.mean(), _ood_adata.X.mean()

    # Dose col
    adata.obs['dose'] = adata.obs[cond_key].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))

    # cov_cond col
    cov_cond = []
    for i in range(adata.shape[0]):
        _name = adata.obs[cov_key][i] + '_' + adata.obs[cond_key][i]
        cov_cond.append(_name)
    adata.obs['cov_cond'] = cov_cond

    # Replace val by valid
    adata.obs[f'split_{stim_name}_{ood_cov}'] = [x.replace('val', 'valid') for x in adata.obs[f'split_{stim_name}_{ood_cov}']]

    # Make DEGs dictionary
    org_ranks = adata.uns[f'rank_genes_groups_{cond_key}'][stim_name]
    new_ranks = {}
    for key in list(org_ranks.keys()):
        new_ranks[f'{key}_{stim_name}'] = org_ranks[key][:50]
    
    adata.uns['rank_genes_groups_cov'] = new_ranks
    
    # Setup anndata
    cpa.CPA.setup_anndata(
        adata, 
        perturbation_key=cond_key,
        control_group=control_name,
        dosage_key='dose',
        categorical_covariate_keys=categorical_attributes,
        is_count_data=True,
        deg_uns_key=f'rank_genes_groups_cov',
        deg_uns_cat_key='cov_cond',
        max_comb_len=1,
        )

    # Model params
    model_params = {
        "n_latent": 64,
        "recon_loss": "nb",
        "doser_type": "linear",
        "n_hidden_encoder": 128,
        "n_layers_encoder": 2,
        "n_hidden_decoder": 512,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": True,
        "dropout_rate_encoder": 0.0,
        "dropout_rate_decoder": 0.1,
        "variational": False,
        "seed": seed_nb,
        }

    # Train params
    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_pretrain_ae": 30,
        "n_epochs_adv_warmup": 50,
        "n_epochs_mixup_warmup": 0,
        "mixup_alpha": 0.0,
        "adv_steps": None,
        "n_hidden_adv": 64,
        "n_layers_adv": 3,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "reg_adv": 20.0,
        "pen_adv": 5.0,
        "lr": 0.0003,
        "wd": 4e-07,
        "adv_lr": 0.0003,
        "adv_wd": 4e-07,
        "adv_loss": "cce",
        "doser_lr": 0.0003,
        "doser_wd": 4e-07,
        "do_clip_grad": True,
        "gradient_clip_value": 1.0,
        "step_size_lr": 10,
    }

    # CPA model
    model = cpa.CPA(
        adata=adata, 
        split_key=f'split_{stim_name}_{ood_cov}',
        train_split='train',
        valid_split='valid',
        test_split='ood',
        **model_params,
               )
    # Train
    model.train(
        max_epochs=2000,
        use_gpu=True, 
        batch_size=512,
        plan_kwargs=trainer_params,
        early_stopping_patience=5,
        check_val_every_n_epoch=5,
        save_path=f'{models_save_path}',
        )

    # Predict
    adata_preds = None
    adata_subset = adata[(adata.obs[cond_key] == control_name) & (adata.obs[cov_key] == ood_cov) & (adata.obs[f'split_{stim_name}_{ood_cov}'] == 'train')].copy()
    ic(adata_subset.X.shape, adata_subset.X.max(), adata_subset.X.min())

    for var_to_predict in vars_to_predict:
        adata_pred = adata_subset.copy()
        adata_pred.obs[f'{cond_key}_org'] = adata_pred.obs[cond_key].copy()
        adata_pred.obs[cond_key] = [var_to_predict] * adata_pred.shape[0]
        
        adata_pred.obs[f'{cond_key}_pred'] = [var_to_predict] * adata_pred.shape[0]
        adata_pred.obs[f'{cond_key}_pred'] = adata_pred.obs[f'{cond_key}_pred'].astype('category')
        
        cov_cond_pred = []
        for i in range(adata_pred.shape[0]):
            _name = adata_pred.obs[cov_key][i] + '_' + adata_pred.obs[f'{cond_key}_pred'][i]
            cov_cond_pred.append(_name)
            
        adata_pred.obs['cov_cond'] = cov_cond_pred

        cpa.CPA.setup_anndata(
                      adata_pred, 
                      perturbation_key=cond_key,
                      control_group=control_name,
                      dosage_key='dose',
                      categorical_covariate_keys=categorical_attributes,
                      is_count_data=True,
                      deg_uns_key=f'rank_genes_groups_cov',
                      deg_uns_cat_key='cov_cond',
                      max_comb_len=1,
                    )
        
        model.predict(adata_pred, batch_size=2048)
        adata_pred.X = adata_pred.obsm['CPA_pred'].copy()
        del adata_pred.obsm['CPA_pred']

        adata_pred = sc.AnnData(
            X=adata_pred.X, 
            obs=adata_pred.obs
            )

        if adata_preds is None:
            
            adata_preds = adata_pred.copy()
        else:
            adata_preds = ad.concat([adata_preds, adata_pred])
            
        #adata_pred.obs['dose'] = [1.0] * adata_pred.shape[0] Or 0
        
    data_stats = {
        'median': data_median,
        'train_mean': train_mean,
        'val_mean': val_mean,
        'ood_mean': ood_mean,
        'train_size': train_size,
        'val_size': val_size,
        'ood_size': ood_size,
        # number of predicted OOD cells stored in `adata_preds`
        'pred_ood': adata_preds.shape[0],
        'adata_subset_shape': adata_subset.shape[0],
        'X_normalization': 'count',
    }
    
    # Replace valid by val
    adata_preds.obs[f'split_{stim_name}_{ood_cov}'] = [x.replace('valid', 'val') for x in adata_preds.obs[f'split_{stim_name}_{ood_cov}']]

    adata_preds.uns.update(data_stats)
    adata_preds.write_h5ad(f'{anndatas_save_path}{ood_cov}_{seed_nb}.h5ad')
    
    # Free memory
    del _train_adata, _val_adata, _ood_adata, adata_preds, adata
    gc.collect()