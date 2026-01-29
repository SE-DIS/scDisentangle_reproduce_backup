import pandas as pd

# SciPy sparse indexing expects mask.nonzero(); pandas Series doesn't have it
if not hasattr(pd.Series, "nonzero"):
    pd.Series.nonzero = lambda self: self.to_numpy().nonzero()
    
import os

from gears import PertData, GEARS
import scanpy as sc
import numpy as np
import anndata as ad
import torch
import random

from icecream import ic
import gc
import json
import pickle

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
    coexpr_cache = os.path.join(
        "data", "norman",
        f"custom_{seed_nb}_None_0.4_20_co_expression_network.csv"
    )
    if os.path.exists(coexpr_cache):
        os.remove(coexpr_cache)
        print('removing cached coexp')
        
    # Set seed
    set_seed(seed_nb)
    print('Starting training for', ood_index, 'with seed', seed_nb)

    # Read original data
    adata_org = sc.read_h5ad(adata_path)
    adata_ctrl = adata_org[adata_org.obs['cond_harm'] == 'ctrl'].copy()

    sc.pp.normalize_total(adata_ctrl, target_sum=adata_org.uns['single_perts_median'])
    sc.pp.log1p(adata_ctrl)

    adata_ctrl.obs['cond_harm_org'] = 'ctrl'
    adata_ctrl.obs['cond_harm_pred'] = 'ctrl'
    
    # Create save directories
    anndatas_save_path = f'predictions/{custom_name}/'
    models_save_path = f'models/{custom_name}/{ood_index}_{seed_nb}/'

    os.makedirs(anndatas_save_path, exist_ok=True)
    os.makedirs(models_save_path, exist_ok=True)
    
    pkl_split_data = {}
    for key in split_dict.keys():
        _conds = split_dict[key]
        _conds = [x+'+ctrl' if not '+' in x else x for x in _conds]
        _conds = [x.replace('ctrl+ctrl', 'ctrl') for x in _conds]
        if key == 'ood':
            key_name = 'test'
        else:
            key_name = key
            
        pkl_split_data[key_name] = _conds

    with open(f'{anndatas_save_path}{ood_index}.pkl', 'wb') as file: 
        pickle.dump(pkl_split_data, file)
    
    # Load data
    pert_data = PertData('./data')
    pert_data.load(data_name = 'norman')
    
    pert_data.prepare_split(
        split = 'custom', 
        split_dict_path=f'{anndatas_save_path}{ood_index}.pkl', 
        seed = seed_nb
    )
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

    # Init model
    gears_model = GEARS(
            pert_data, device = 'cuda:0',
            weight_bias_track = False, 
            proj_name = 'pertnet', 
            exp_name = f'pertnet_{custom_name}_{ood_index}'
        )
    
    gears_model.model_initialize(hidden_size = 64)

    # Train
    gears_model.train(epochs = 20, lr = 1e-3)

    # Save model
    gears_model.save_model(models_save_path)

    # Load model
    gears_model.load_pretrained(models_save_path)

    oods_split = [x.split('+') for x in vars_to_predict]
    preds = gears_model.predict_sc(oods_split)
    preds_mean = gears_model.predict(oods_split)

    data_stats = {
        'X_normalization': 'median'
    }
    
    for ood_label in preds.keys():
        pert_name = ood_label.replace('_', '+')
        pred_X = preds[ood_label]

        adata_pred = sc.AnnData(X=pred_X)
        adata_pred.obs['cond_harm_pred'] = pert_name
        adata_pred.obs['cond_harm'] = pert_name
        adata_pred.obs['cond_harm_org'] = 'ctrl'
        adata_pred.obs['sc_cell_ids'] = adata_ctrl.obs['sc_cell_ids'].copy()
        adata_pred.var_names = adata_org.var_names.copy()
        
        adata_pred = ad.concat([adata_pred, adata_ctrl.copy()])
        adata_pred.uns.update(data_stats)
        adata_pred.uns[f'{ood_label}_mean'] = preds_mean[ood_label]
        adata_pred.write_h5ad(f'{anndatas_save_path}{ood_label}.h5ad')
        
        del adata_pred
        gc.collect()

    del adata_org, adata_ctrl
    
    gc.collect()
    with open(f'{anndatas_save_path}{ood_index}.json', 'w') as f:
        json.dump({'ood_index':int(ood_index)}, f)