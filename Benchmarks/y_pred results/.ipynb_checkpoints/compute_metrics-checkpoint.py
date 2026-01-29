import os
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

import metrics_tools as mt

def compute_metrics(
    data_name,
    yaml_name,
    methods,
    cov_key,
    cond_key,
    control_name,
    stim_name,
    degs_key,
    ood_covs=None,
    degs_stim=None,
    seed_list=None,
    custom_name='',
):
    """
    Compute correlation and distance-based metrics for counterfactual predictions.
    """
    # Load preprocessed dataset
    adata_org = sc.read_h5ad(
        f"../../Datasets/preprocessed_datasets/{data_name.lower()}.h5ad"
    )
    if not isinstance(adata_org.X, np.ndarray):
        adata_org.X = adata_org.X.toarray()
        
    if seed_list is None:
        seed_list = list(range(1, 11))
        
    if degs_stim is None:
        degs_stim = stim_name
    
    # OOD covariates: default to all unique covs in the dataset
    if ood_covs is None:
        ood_covs = adata_org.obs[cov_key].unique().tolist()

    for ood_cov in tqdm(ood_covs):
        print(f"Computing metrics for {ood_cov}")
        adata = adata_org.copy()

        # DEGs
        degs = adata_org.uns[degs_key][degs_stim][ood_cov]
        degs_indices = [adata_org.var_names.get_loc(g) for g in degs]

        # Compute train median exactly as in training scripts: use the train split
        split_key = f"split_{stim_name}_{ood_cov}"

        train_mask = adata.obs[split_key] == "train"
        _sums = adata.X[train_mask].sum(axis=1, keepdims=True)
       
        data_median = np.median(_sums)

        # Normalize ground-truth data to median library size then log1p
        sc.pp.normalize_total(adata, target_sum=data_median)
        sc.pp.log1p(adata)

        # GT stim and GT ctrl for this OOD covariate
        true_stim = adata[
            (adata.obs[cov_key] == ood_cov) & (adata.obs[cond_key] == stim_name) & (adata.obs[split_key] == 'ood')
        ].copy()
        true_ctrl = adata[
            (adata.obs[cov_key] == ood_cov) & (adata.obs[cond_key] == control_name) & (adata.obs[split_key] == 'train')
        ].copy()

        for seed_nb in seed_list:
            for method_name in methods:
                # Path to prediction AnnData
                if method_name == "SCDISENTANGLE":
                    preds_path = (
                        f"../{method_name}/{data_name}/predictions/{yaml_name}/"
                        f"{ood_cov}_{seed_nb}.h5ad"
                    )
                    
                else:
                    preds_path = (
                        f"../{method_name}/{data_name}/predictions/{ood_cov}_{seed_nb}.h5ad"
                    )

                if not os.path.isfile(preds_path) :
                    print("Skipping, missing:", preds_path)
                    continue
                
                adata_pred = sc.read_h5ad(preds_path)                
                
                if method_name != 'SCDISINFACT':
                    print(method_name, adata_pred.uns['median'], data_median)
                    assert np.isclose(adata_pred.uns['median'], data_median, rtol=1e-6, atol=1e-3)
                    
                adata_pred = adata_pred[adata_pred.obs[split_key] == "train"]
                
                # Normalize predicted data
                if adata_pred.uns['X_normalization'] == 'count':
                    print(method_name, 'normalizing by count')
                    sc.pp.normalize_total(adata_pred, target_sum=data_median)
                    sc.pp.log1p(adata_pred)

                # Predicted stim / ctrl for this OOD covariate
                pred_stim = adata_pred[
                    (adata_pred.obs[cov_key] == ood_cov)
                    & (adata_pred.obs[f'{cond_key}_pred'] == stim_name)
                ].copy()
                pred_ctrl = adata_pred[
                    (adata_pred.obs[cov_key] == ood_cov)
                    & (adata_pred.obs[f'{cond_key}_pred'] == control_name)
                ].copy()

                pert_mean = adata[
                    (adata.obs[f'split_{stim_name}_{ood_cov}'] == 'train') & \
                     (adata.obs[cov_key] != ood_cov) & \
                     (adata.obs[cond_key] == stim_name)
                     ].X.mean(axis=0)
                
                assert pred_stim.shape[0] == pred_ctrl.shape[0]
                # Compute metrics
                all_metrics = {}
                        
                # Correlation metrics: nested dict (metric -> {n_degs: value})
                corr_metrics = mt.get_correlations(
                    _pred_stim=pred_stim,
                    _true_stim=true_stim,
                    _pred_ctrl=pred_ctrl,
                    _true_ctrl=true_ctrl,
                    _pert_mean=pert_mean,
                    degs_indices=degs_indices,
                    degs_list=[200, 100, 50, 20, 10],
                )
                all_metrics.update(corr_metrics)

                # Distance metrics: same nested structure
                dist_metrics = mt.get_distances(
                    _pred_stim=pred_stim,
                    _true_stim=true_stim,
                    _pred_ctrl=pred_ctrl,
                    _true_ctrl=true_ctrl,
                    degs_indices=degs_indices,
                    degs_list=[200, 100, 50, 20, 10],
                )
                all_metrics.update(dist_metrics)
                
                # Subset true_ctrl to only include train CTRL cells!
                true_ctrl = true_ctrl[true_ctrl.obs[split_key] == "train"].copy()

                # Here sort pred_stim to have same order as true_ctrl (using obs['sc_cell_ids'])
                id_to_idx = {
                    cid: i for i, cid in enumerate(pred_stim.obs['sc_cell_ids'])
                }
                order = [id_to_idx[cid] for cid in true_ctrl.obs['sc_cell_ids']]
                
                pred_stim = pred_stim[order].copy()
                #pred_ctrl = pred_ctrl[order].copy() 

                assert np.array_equal(
                    pred_stim.obs['sc_cell_ids'].to_numpy(), 
                    true_ctrl.obs['sc_cell_ids'].to_numpy()
                    )
                
                # Single-cell preservation metric
                sc_sim_nested = mt.get_sc_similarity(
                    _pred_stim=pred_stim,
                    _true_ctrl=true_ctrl,
                )
                sc_sim_values = sc_sim_nested['All']
                sc_degs = [200, 100, 50, 20, 10]
                for metric_name, value in sc_sim_values.items():
                    all_metrics[metric_name] = {n: value for n in sc_degs}

                # Save metrics: rows = metric names, columns = DEG subset sizes
                metrics_df = pd.DataFrame(all_metrics).T
                metrics_df.index.rename("Metric", inplace=True)

                save_path = f"results/{data_name}/{method_name}"
                os.makedirs(save_path, exist_ok=True)
                
                metrics_df.to_csv(f"{save_path}/{custom_name}{ood_cov}_{seed_nb}.csv")

def compute_metrics_baselines(
    data_name,
    methods,
    cov_key,
    cond_key,
    control_name,
    stim_name,
    degs_key,
    ood_covs=None,
    degs_stim=None,
    seed_list=None,
    custom_name=''
):
    """
    Compute correlation and distance-based metrics for counterfactual predictions.
    """
    if seed_list is None:
        seed_list = list(range(1, 11))
    # Load preprocessed dataset
    adata_org = sc.read_h5ad(
        f"../../Datasets/preprocessed_datasets/{data_name.lower()}.h5ad"
    )

    if not isinstance(adata_org.X, np.ndarray):
        adata_org.X = adata_org.X.toarray()
        
    if degs_stim is None:
        degs_stim = stim_name
    
    
    # OOD covariates: default to all unique covs in the dataset
    if ood_covs is None:
        ood_covs = adata_org.obs[cov_key].unique().tolist()

    for ood_cov in tqdm(ood_covs):
        print(f"Computing metrics for {ood_cov}")
        adata = adata_org.copy()

        # DEGs
        degs = adata_org.uns[degs_key][degs_stim][ood_cov]
        degs_indices = [adata_org.var_names.get_loc(g) for g in degs]

        # Compute train median exactly as in training scripts: use the train split
        split_key = f"split_{stim_name}_{ood_cov}"

        train_mask = adata.obs[split_key] == "train"
        _sums = adata.X[train_mask].sum(axis=1, keepdims=True)
       
        data_median = np.median(_sums)

        # Normalize ground-truth data to median library size then log1p
        sc.pp.normalize_total(adata, target_sum=data_median)
        sc.pp.log1p(adata)
                        
        # GT stim and GT ctrl for this OOD covariate
        true_stim = adata[
            (adata.obs[cov_key] == ood_cov) & (adata.obs[cond_key] == stim_name) & (adata.obs[split_key] == 'ood')
        ].copy()
        true_ctrl = adata[
            (adata.obs[cov_key] == ood_cov) & (adata.obs[cond_key] == control_name) & (adata.obs[split_key] == 'train')
        ].copy()
        
        adata_pred = adata[adata.obs[split_key] == "train"].copy()    
        for method_name in methods:
                
                # Predicted stim / ctrl for this OOD covariate
                if method_name == 'Pert Mean':
                    pred_stim = adata_pred[
                        (adata_pred.obs[cov_key] != ood_cov)
                        & (adata_pred.obs[cond_key] == stim_name) 
                    ].copy()

                elif method_name == 'Ctrl Mean':
                    pred_stim = adata_pred[
                        (adata_pred.obs[cov_key] == ood_cov)
                        & (adata_pred.obs[cond_key] == control_name)
                    ].copy()

                else:
                    raise ValueError('method_name should be Pert Mean or Ctrl Mean')
                    
                pred_ctrl = adata_pred[
                    (adata_pred.obs[cov_key] == ood_cov) & (adata_pred.obs[cond_key] == control_name)
                    ].copy()

                # Compute metrics
                all_metrics = {}
                        
                # Correlation metrics: nested dict (metric -> {n_degs: value})
                corr_metrics = mt.get_correlations(
                    _pred_stim=pred_stim,
                    _true_stim=true_stim,
                    _pred_ctrl=pred_ctrl,
                    _true_ctrl=true_ctrl,
                    degs_indices=degs_indices,
                    degs_list=[200, 100, 50, 20, 10],
                )
                all_metrics.update(corr_metrics)

                # Distance metrics: same nested structure
                dist_metrics = mt.get_distances(
                    _pred_stim=pred_stim,
                    _true_stim=true_stim,
                    _pred_ctrl=pred_ctrl,
                    _true_ctrl=true_ctrl,
                    degs_indices=degs_indices,
                    degs_list=[200, 100, 50, 20, 10],
                )
                all_metrics.update(dist_metrics)
                
                # Save metrics: rows = metric names, columns = DEG subset sizes
                metrics_df = pd.DataFrame(all_metrics).T
                metrics_df.index.rename("Metric", inplace=True)

                save_path = f"results/{data_name}/{method_name}"
                os.makedirs(save_path, exist_ok=True)

                for seed_nb in seed_list:
                    metrics_df.to_csv(f"{save_path}/{custom_name}{ood_cov}_{seed_nb}.csv")