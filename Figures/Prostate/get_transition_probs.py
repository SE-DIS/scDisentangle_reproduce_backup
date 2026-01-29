import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import scvelo as scv
import os
from tqdm import tqdm

from scdisentangle.train.tools import get_trainer, set_seed
import local_tools as lt

def cluster_transition_matrix(adata, tm, cluster_key="cell_type"):
    """
  
    """
    if issparse(tm):
        tm = tm.A

    clusters = adata.obs[cluster_key].values
    unique = np.unique(clusters)
    n = len(unique)

    # map cluster name -> indices of cells
    idxs = {c: np.where(clusters == c)[0] for c in unique}

    C = np.zeros((n, n), dtype=float)
    for i, c_src in enumerate(unique):
        src_idx = idxs[c_src]
        n_src = len(src_idx)
        for j, c_tgt in enumerate(unique):
            tgt_idx = idxs[c_tgt]
            total = tm[src_idx][:, tgt_idx].sum()
            C[i, j] = total / max(n_src, 1)

    # row-normalize so each source cluster’s row sums to 1
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    C = C / row_sums

    return pd.DataFrame(C, index=unique, columns=unique)

def get_velocity_adata(
    adata_inp,
    current_time,
    future_time,
    current_time_method, # 'rec', 'gt', 'pred'
    input_cells, # 'current_time', 'T00' 
    ):

    assert input_cells in ['current_time', 'T00']

    # Predict future state

    # From current_time cells
    if input_cells == 'current_time':
        future_inp =  adata_inp[adata_inp.obs['time'] == current_time].copy()

    # OR from T0 cells
    elif input_cells == 'T00':
        future_inp = adata_inp[adata_inp.obs['time'] == 'T00'].copy()

    future_pred = trainer.predict(
        future_inp.copy(),
        counterfactual_dict={'time':future_time},
        bs=256
    )
    
    assert current_time_method in ['rec', 'gt', 'pred']

    # Get current time cells

    # By just reconstructing them
    if current_time_method == 'rec':
        current_inp = adata_inp[adata_inp.obs['time'] == current_time].copy()

        current_pred = trainer.predict(
            current_inp.copy(),
            counterfactual_dict={'time':current_time},
            bs=256
        )

    # Or by predicting them from T0 cells
    elif current_time_method == 'pred':
        current_inp = adata_inp[adata_inp.obs['time'] == 'T00'].copy()
        
        current_pred = trainer.predict(
            current_inp.copy(),
            counterfactual_dict={'time':current_time},
            bs=256
        )

    # Or by taking them from gt data
    elif current_time_method == 'gt':
        current_pred = adata_inp[adata_inp.obs['time'] == current_time].copy()


    # Still all of X, spliced and unspliced are in counts
    adata = current_pred.copy()
    adata.layers['spliced'] = adata.X.copy()
    adata.layers['unspliced'] = future_pred.X.copy()

    # Compute velocity
    velocity = adata.layers['unspliced'] - adata.layers['spliced']
    adata.obsm['computed_velocity'] = velocity

    return adata

for rand_idx in tqdm(range(1)):
    seed_nb = rand_idx
    set_seed(seed_nb)
    
    yaml_path = 'configs/prostate_counterfactual.yaml'
    
    trainer = get_trainer(yaml_path, wandb_log=False, seed_nb=seed_nb)
    
    weights_path = 'counterfactual_weights/MIG_BINNED_dis_latent_stack_time_train'
    trainer.load_weights(weights_path)
    
    ctypes_to_filter = [
        'Epi_SV_Ionocyte', 'Epi_SV_Luminal', 'Epi_SV_Basal',
        'SymDoublet_Epi_Imm', 'SymDoublet_Str_Epi', 'SymDoublet_Str_Imm'
    ]
    
    cell_types_to_keep = ['Epi_Luminal_1', 'Epi_Luminal_2Psca', 'Epi_Luminal_3Foxi1', 'Epi_Basal_1']
    
    adata_org = trainer.dataset.data[~trainer.dataset.data.obs['predType'].isin(ctypes_to_filter)].copy()

    np.random.seed(rand_idx)
    adata_org = adata_org[adata_org.obs['predType'].isin(cell_types_to_keep)]
    
    adata_org.obs['cell_type'] = adata_org.obs['predType'].copy()

    # k = int(0.75 * adata_org.n_obs)
    # print('size before subsetting', adata_org.shape[0])
    # adata_org = adata_org[np.random.choice(adata_org.shape[0], size=k, replace=False)]
    # print('size before subsetting', adata_org.shape[0])

    # from icecream import ic
    # print(adata_org.X.mean())
        
    _adata_velocity = get_velocity_adata(
        adata_inp=adata_org.copy(),
        current_time='T04_Cast_Day28',
        future_time='T10_Regen_Day28',
        current_time_method='pred',
        input_cells='T00' 
        )
    
    scv.pp.filter_and_normalize(
        _adata_velocity, 
        min_shared_counts=20, 
        n_top_genes=2000,
        enforce=True
    )

    for idx in range(10, 50):

        if idx < 10:
            np.random.choice(_adata_velocity.shape[0], size=_adata_velocity.shape[0], replace=True)
            continue
        #k = int(0.75 * _adata_velocity.n_obs)
        adata_velocity = _adata_velocity[np.random.choice(_adata_velocity.shape[0], size=_adata_velocity.shape[0], replace=True)].copy()
        
        scv.pp.neighbors(adata_velocity)
        
        adata_velocity.layers['velocity'] = adata_velocity.layers['unspliced'] - adata_velocity.layers['spliced']
        adata_velocity.obs['clusters'] = adata_velocity.obs['cell_type']
        scv.tl.velocity_graph(adata_velocity)
        
        mtx = scv.utils.get_transition_matrix(adata_velocity)
        mtx = scv.utils.get_transition_matrix(adata_velocity)
        cluster_tm = cluster_transition_matrix(adata_velocity, mtx, cluster_key="cell_type")
        cluster_tm
        
        epi = ["Epi_Luminal_1", "Epi_Luminal_2Psca", 'Epi_Luminal_3Foxi1', 'Epi_Basal_1']  # adjust to your labels
        sub = cluster_tm.loc[epi, epi].copy()
        for c in epi:
            if c in sub.columns:
                sub.loc[c, c] = 0.0  # remove self-transition
        avg_epi_transition = float(sub.values.mean())
        avg_epi_transition
        
        with open('average_transition_regeneration.txt', 'a') as f:
            f.write(f'{avg_epi_transition}\n')
    
        del adata_velocity
        #del adata_org
    
        import gc
        gc.collect()