import os
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import json

import metrics_tools as mt
from scdisentangle.train.CustomIterations import CustomIterations
import gc

def compute_metrics(
    data_name,
    custom_name,
    cond_key,
    control_name,
    methods,
    ood_labels,
    degs_key,
    scenario,
):
    """
    Compute correlation and distance-based metrics for counterfactual predictions.
    """

    with open(
        f"../../Datasets/preprocessed_datasets/per_pert_splits_{scenario}.json",
        'r'
        
    ) as f:
        per_pert_splits = json.load(f)
        
    # Load preprocessed dataset
    adata_org = sc.read_h5ad(
        f"../../Datasets/preprocessed_datasets/{data_name.lower()}.h5ad"
    )
    if not isinstance(adata_org.X, np.ndarray):
        adata_org.X = adata_org.X.toarray()

    for ood_l in tqdm(ood_labels):
        train_subset = per_pert_splits[ood_l]
        
        pert1, pert2 = ood_l.split('+')
        rg = adata_org.uns[degs_key][ood_l]
        degs_indices = [adata_org.var_names.get_loc(g) for g in rg]

        adata = adata_org.copy()
       
        # Normalize ground-truth data to median library size then log1p
        sc.pp.normalize_total(adata, target_sum=adata.uns['single_perts_median'])
        sc.pp.log1p(adata)

        # GT stim and GT ctrl for this OOD covariate
        true_stim = adata[
            adata.obs[cond_key] == ood_l
        ].copy()
        true_ctrl = adata[
            adata.obs[cond_key] == control_name
        ].copy()

        for method_name in methods:
            # Path to prediction AnnData
            if method_name == 'GEARS':
                preds_path = (
                f"../{method_name}/{data_name}/predictions/{custom_name}/{ood_l.replace('+', '_')}.h5ad"
            )
            else:
                preds_path = (
                    f"../{method_name}/{data_name}/predictions/{custom_name}/{ood_l}.h5ad"
                )
            
            if not os.path.isfile(preds_path) :
                print("Skipping, missing:", preds_path)
                continue
            
            adata_pred = sc.read_h5ad(preds_path)                
                        
            # Normalize predicted data
            if adata_pred.uns['X_normalization'] == 'count':
                print(method_name, 'normalizing by count')
                sc.pp.normalize_total(adata_pred, target_sum=adata.uns['single_perts_median'])
                sc.pp.log1p(adata_pred)

            # Predicted stim / ctrl for this OOD covariate
                
            pred_stim = adata_pred[
                (adata_pred.obs[f'{cond_key}_pred'] == ood_l)
            ].copy()
            
            pred_ctrl = adata_pred[
                (adata_pred.obs[f'{cond_key}_pred'] == control_name)
            ].copy()

            pert_mean = adata[adata.obs[f'{cond_key}'].isin(train_subset)].X.mean(axis=0)
            assert len(adata[adata.obs[f'{cond_key}'].isin(train_subset)]) > 10

            if method_name == 'SCDISENTANGLE':
                # Calibrate
                print('Calibrating')
                
                calibration_avg = CustomIterations().calibrate_predictions(
                    adata_pred=adata_pred,
                    adata_gt=adata,
                    cond_key=cond_key,
                    target_perturbation=ood_l
                )
                pred_stim = pred_stim.copy()
                calibrated_X = pred_stim.X + calibration_avg
                pred_stim.X = calibrated_X
                
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

            true_pert1 = adata[adata.obs[cond_key]==pert1].copy()
            true_pert2 = adata[adata.obs[cond_key]==pert2].copy()
        
            specificity_metrics = mt.get_specificity_score(
                _pred_stim=pred_stim,
                _true_stim=true_stim,
                _true_ctrl=true_ctrl,
                _true_pert1=true_pert1,
                _true_pert2=true_pert2,
                degs_indices=degs_indices,
                degs_list=[200, 100, 50, 20, 10]
            )
            all_metrics.update(specificity_metrics)

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

            centroid_acc = mt.get_centroid_accuracy(
                _pred_stim=pred_stim,
                adata_org=adata.copy(),
                pert_name=ood_l 
            )
            
            c_values = centroid_acc['All']
            c_degs = [200, 100, 50, 20, 10]
            for metric_name, value in c_values.items():
                all_metrics[metric_name] = {n: value for n in c_degs}
            
            if method_name != 'GEARS':
                # sort pred_stim to have same order as true_ctrl using obs['sc_cell_ids']
                id_to_idx = {
                    cid: i for i, cid in enumerate(pred_stim.obs['sc_cell_ids'])
                }
                order = [id_to_idx[cid] for cid in true_ctrl.obs['sc_cell_ids']]
                
                pred_stim = pred_stim[order].copy()
    
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

            save_path = f"results/{data_name}/{custom_name}/{method_name}"
            os.makedirs(save_path, exist_ok=True)
            metrics_df.to_csv(f"{save_path}/{ood_l}.csv")

def compute_metrics_baselines(
    data_name,
    methods,
    cond_key,
    control_name,
    degs_key,
    ood_labels,
    scenario,
    custom_name,
):
    """
    Compute correlation and distance-based metrics for counterfactual predictions.
    """
    with open(
        f"../../Datasets/preprocessed_datasets/per_pert_splits_{scenario}.json",
        'r'
        
    ) as f:
        per_pert_splits = json.load(f)
   
    # Load preprocessed dataset
    adata_org = sc.read_h5ad(
        f"../../Datasets/preprocessed_datasets/{data_name.lower()}.h5ad"
    )
    
    if not isinstance(adata_org.X, np.ndarray):
        adata_org.X = adata_org.X.toarray()

    for ood_l in tqdm(ood_labels):
        train_subset = per_pert_splits[ood_l] + ['ctrl']
        
        rg = adata_org.uns[degs_key]
        ranked_genes = rg[ood_l]
        degs_indices = [adata_org.var_names.get_loc(g) for g in ranked_genes]

        print(f"Computing metrics for {ood_l}")
        adata = adata_org.copy()

        # Normalize ground-truth data to median library size then log1p
        sc.pp.normalize_total(adata, target_sum=adata.uns['single_perts_median'])
        sc.pp.log1p(adata)
                        
        # GT stim and GT ctrl for this OOD covariate
        true_stim = adata[
           adata.obs[cond_key] == ood_l
        ].copy()
        true_ctrl = adata[
          adata.obs[cond_key] == control_name
        ].copy()

        adata_pred = adata[adata.obs[cond_key].isin(train_subset)].copy()
        all_perts = adata_pred.obs[cond_key].unique().tolist()
        pert1, pert2 = ood_l.split('+')
        
        for method_name in methods:
                
                # Predicted stim / ctrl for this OOD covariate
                if method_name == 'Pert Mean':
                    pred_stim = adata_pred[
                        adata_pred.obs[cond_key] != control_name
                    ].copy()

                elif method_name == 'Ctrl Mean':
                    pred_stim = adata_pred[
                        adata_pred.obs[cond_key] == control_name
                    ].copy()

                elif method_name == 'Match Mean':
                    
                    
                    pert1_perts = [x for x in all_perts if f'{pert1}' in x.split('+')]
                    pert2_perts = [x for x in all_perts if f'{pert2}' in x.split('+')]

                    pert1_mean = adata_pred[adata_pred.obs[cond_key].isin(pert1_perts)].X.mean(axis=0)
                    pert2_mean = adata_pred[adata_pred.obs[cond_key].isin(pert2_perts)].X.mean(axis=0)

                    pred_stim = (pert1_mean + pert2_mean)/2
                    pred_stim = sc.AnnData(pred_stim.reshape(1, -1))

                    print('Match mean sizes', len(pert1_perts), len(pert2_perts))

                else:
                    raise ValueError('method_name should be Pert Mean or Ctrl Mean')
                    
                pred_ctrl = adata_pred[
                   adata_pred.obs[cond_key] == control_name
                    ].copy()

                # Compute metrics
                all_metrics = {}
                        
                # correlation metrics
                corr_metrics = mt.get_correlations(
                    _pred_stim=pred_stim,
                    _true_stim=true_stim,
                    _pred_ctrl=pred_ctrl,
                    _true_ctrl=true_ctrl,
                    degs_indices=degs_indices,
                    degs_list=[200, 100, 50, 20, 10],
                )
                all_metrics.update(corr_metrics)
                
                # Distance metrics
                dist_metrics = mt.get_distances(
                    _pred_stim=pred_stim,
                    _true_stim=true_stim,
                    _pred_ctrl=pred_ctrl,
                    _true_ctrl=true_ctrl,
                    degs_indices=degs_indices,
                    degs_list=[200, 100, 50, 20, 10],
                )
                all_metrics.update(dist_metrics)
            
                true_pert1 = adata[adata.obs[cond_key]==pert1].copy()
                true_pert2 = adata[adata.obs[cond_key]==pert2].copy()
            
                specificity_metrics = mt.get_specificity_score(
                    _pred_stim=pred_stim,
                    _true_stim=true_stim,
                    _true_ctrl=true_ctrl,
                    _true_pert1=true_pert1,
                    _true_pert2=true_pert2,
                    degs_indices=degs_indices,
                    degs_list=[200, 100, 50, 20, 10]
                )
                all_metrics.update(specificity_metrics)

                centroid_acc = mt.get_centroid_accuracy(
                    _pred_stim=pred_stim,
                    adata_org=adata.copy(),
                    pert_name=ood_l 
                )

                c_values = centroid_acc['All']
                c_degs = [200, 100, 50, 20, 10]
                for metric_name, value in c_values.items():
                    all_metrics[metric_name] = {n: value for n in c_degs}
                    
                # Save metrics: rows = metric names, columns = DEG subset sizes
                metrics_df = pd.DataFrame(all_metrics).T
                metrics_df.index.rename("Metric", inplace=True)

                save_path = f"results/{data_name}/{custom_name}/{method_name}"
                os.makedirs(save_path, exist_ok=True)

                
                metrics_df.to_csv(f"{save_path}/{ood_l}.csv")
                
        del adata_pred, pred_stim, true_stim, true_ctrl, pred_ctrl
        gc.collect()