import os

import cpa
import scanpy as sc
import numpy as np
import anndata as ad
import torch
import random

from icecream import ic
import gc
import json

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
    ood_index,
    seed_nb,
    split_dict,
    vars_to_predict,
    custom_name,
    ):

    # Set seed
    set_seed(seed_nb)
    
    print('Starting training for', ood_index, 'with seed', seed_nb)
                          
    # Create save directories
    anndatas_save_path = f'predictions/{custom_name}/'
    models_save_path = f'models/{custom_name}/{ood_index}_{seed_nb}/'

    os.makedirs(anndatas_save_path, exist_ok=True)
    os.makedirs(models_save_path, exist_ok=True)     

    # Read adata
    adata = sc.read_h5ad(adata_path)

    # Create train/val/ood splits
    train_mask = adata.obs['cond_harm'].isin(split_dict['train'])
    val_mask = adata.obs['cond_harm'].isin(split_dict['val'])
    ood_mask = adata.obs['cond_harm'].isin(split_dict['ood'])

    adata.obs['split'] = None
    adata.obs['split'][train_mask] = 'train'
    adata.obs['split'][val_mask] = 'valid'
    adata.obs['split'][ood_mask] = 'ood'
    
    print(adata.obs['split'].value_counts())
    
    # Set to counts:
    adata.X = adata.layers['counts'].copy()
    try:
        adata.X = adata.X.toarray()
    except:
        pass

    # Dose col
    adata.obs['dose_value'] = adata.obs['cond_harm'].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))

    # cov_cond col
    cov_cond = []
    for i in range(adata.shape[0]):
        _name = adata.obs['cell_type'][i] + '_' + adata.obs['condition'][i]
        cov_cond.append(_name)
    adata.obs['cov_cond'] = cov_cond

    # DEGs
    adata.uns['rank_genes_groups_cov'] = {f'K562_{k}':v[:200] for k,v in adata.uns['rank_genes_groups'].items()}
    
    # Setup anndata
    cpa.CPA.setup_anndata(adata, 
        perturbation_key='cond_harm',
        control_group='ctrl',
        dosage_key='dose_value',
        categorical_covariate_keys=['cell_type'],
        is_count_data=True,
        deg_uns_key='rank_genes_groups_cov',
        deg_uns_cat_key='cov_cond',
        max_comb_len=2,
        )

    # Parameters
    model_params = {
        "n_latent": 32,
        "recon_loss": "nb",
        "doser_type": "linear",
        "n_hidden_encoder": 256,
        "n_layers_encoder": 4,
        "n_hidden_decoder": 256,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": False,
        "dropout_rate_encoder": 0.2,
        "dropout_rate_decoder": 0.0,
        "variational": False,
        "seed": 8206,
    }

    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_adv_warmup": 50,
        "n_epochs_mixup_warmup": 10,
        "n_epochs_pretrain_ae": 10,
        "mixup_alpha": 0.1,
        "lr": 0.0001,
        "wd": 3.2170178270865573e-06,
        "adv_steps": 3,
        "reg_adv": 10.0,
        "pen_adv": 20.0,
        "adv_lr": 0.0001,
        "adv_wd": 7.051355554517135e-06,
        "n_layers_adv": 2,
        "n_hidden_adv": 128,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "step_size_lr": 25,
        "do_clip_grad": False,
        "adv_loss": "cce",
        "gradient_clip_value": 5.0,
    }

    # Init model
    model = cpa.CPA(
        adata=adata, 
        split_key='split',        
        train_split='train',
        valid_split='valid',
        test_split='ood',
        **model_params,
               )

    # Train
    model.train(max_epochs=2000,
            use_gpu=True, 
            batch_size=2048,
            plan_kwargs=trainer_params,
            early_stopping_patience=5,
            check_val_every_n_epoch=5,
            save_path=f'{models_save_path}',
           )


    # Compute data stats (train / val / ood)
    _train_adata = adata[adata.obs['split'] == 'train'].copy()
    _val_adata = adata[adata.obs['split'] == 'valid'].copy()
    _ood_adata = adata[adata.obs['split'] == 'ood'].copy()

    # Compute median
    #_sums = _train_adata.X.sum(axis=1, keepdims=True)
    #data_median = np.median(_sums)

    # Compute means
    train_size, val_size, ood_size = _train_adata.shape[0], _val_adata.shape[0], _ood_adata.shape[0]
    train_mean, val_mean, ood_mean = _train_adata.X.mean(), _val_adata.X.mean(), _ood_adata.X.mean()
    
    # Predict oods
    adata_subset = adata[adata.obs['cond_harm'] == 'ctrl']
    
    for var_to_predict in ['ctrl'] + vars_to_predict:
        adata_pred = adata_subset.copy()

        # Make cond_harm_org
        adata_pred.obs['cond_harm_org'] = adata_pred.obs['cond_harm'].copy()

        # Set cond_harm and cond_harm_pred to var_to_predict
        adata_pred.obs['cond_harm_pred'] = [var_to_predict] * adata_pred.shape[0]
        adata_pred.obs['cond_harm'] = [var_to_predict] * adata_pred.shape[0]

        # Set cov_cond to cell_type + _ + var_to_predict
        cov_cond_pred = []
        for i in range(adata_pred.shape[0]):
            _name = adata_pred.obs['cell_type'][i] + '_' + adata_pred.obs['cond_harm'][i]
            cov_cond_pred.append(_name)

        adata_pred.obs['cov_cond'] = cov_cond_pred

        # Set dose value
        adata_pred.obs['dose_value'] = adata_pred.obs['cond_harm'].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))

        # Setup anndata
        cpa.CPA.setup_anndata(
            adata_pred, 
            perturbation_key='cond_harm',
            control_group='ctrl',
            dosage_key='dose_value',
            categorical_covariate_keys=['cell_type'],
            is_count_data=True,
            deg_uns_key='rank_genes_groups_cov',
            deg_uns_cat_key='cov_cond',
            max_comb_len=2,
                )

        # Predict
        model.predict(adata_pred, batch_size=2048)

        # Set X to preds
        adata_pred.X = adata_pred.obsm['CPA_pred'].copy()


        del adata_pred.obsm['CPA_pred']
        del adata_pred.layers['counts']

        # Infos
        data_stats = {
            #'median': data_median,
            'train_mean': train_mean,
            'val_mean': val_mean,
            'ood_mean': ood_mean,
            'train_size': train_size,
            'val_size': val_size,
            'ood_size': ood_size,
            'pred_ood': adata_pred.shape[0],
            'adata_subset_shape': adata_subset.shape[0],
            'X_normalization': 'count',
        }
        
        # Replace valid by val
        adata_pred.obs['split'] = [x.replace('valid', 'val') for x in adata_pred.obs['split']]

        if var_to_predict == 'ctrl':
            adata_pred_ctrl = adata_pred.copy()
        else:
            adata_pred = ad.concat([adata_pred, adata_pred_ctrl])
            adata_pred.uns.update(data_stats)
            adata_pred.write_h5ad(f'{anndatas_save_path}{var_to_predict}.h5ad')

    # Free memory
    del _train_adata, _val_adata, _ood_adata, adata_pred, adata
    gc.collect()
    with open(f'{anndatas_save_path}{ood_index}.json', 'w') as f:
        json.dump({'ood_index':int(ood_index)}, f)