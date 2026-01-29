from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
import scanpy as sc
import numpy as np
import pandas as pd
import os

correlation_metrics = [
    'Pearson Mean', 'Pearson Mean Delta', 'R2 Mean', 'R2 Mean Delta',
    'Perturbed Reference Pearson Delta', 'Single-cell Identity preservation',
    'Perturbed Reference R2 Delta', 'Centroid Accuracy'
]

distance_metrics = [
    'RMSE Mean', 'Normalized RMSE Mean',
    'Wasserstein', 'Relative Wasserstein Distance'
]

def compute_pvalues_by_context(
    metrics_dict,
    metric_name,
    run_ids,
    anchor="scDisentangle",
    higher_is_better=None,
    alpha=0.05,
):
    from scipy.stats import mannwhitneyu, combine_pvalues
    """
    Unpaired per context test (Mann-Whitney U, one-sided) & aggregate contexts using fisher

    metrics_dict: {method: {degs_group: [values aligned to run_ids]}}
    run_ids: ['B_1','B_2',..]
    anchor: method to compare against others
    higher_is_better: None, True, False
    alpha: significance threshold
    
    Returns: p_values, stars
    """
    if higher_is_better is None:
        if metric_name in correlation_metrics:
            higher_is_better = True
        elif metric_name in distance_metrics:
            higher_is_better = False
        else:
            raise ValueError(f'higher_is_better unclear for {metric_name}')
            
    run_ids = list(run_ids)
    contexts = np.array([rid.rsplit("_", 1)[0] for rid in run_ids])
    
    groups = metrics_dict[anchor].keys()
    p_values = {}
    stars = {}

    for group in groups:
        a = np.array(metrics_dict[anchor][group])
        if len(a) != len(run_ids) or not np.isfinite(a).all():
            raise ValueError('Length mismatch')

        # pick competitor by best median
        cand = []
        for m, sub in metrics_dict.items():
            if m == anchor: 
                continue
            b = np.array(metrics_dict[m][group])
            if b.shape != a.shape or not np.isfinite(b).all():
                continue
            cand.append((m, float(np.median(b))))
        if not cand:
            raise ValueError('cand empty')
            
        competitor = max(cand, key=lambda x: x[1])[0] if higher_is_better else min(cand, key=lambda x: x[1])[0]

        b = np.array(metrics_dict[competitor][group], dtype=float)

        # PER context p-vals
        p_ctx = []
        for ctx in np.unique(contexts):
            idx = np.where(contexts == ctx)[0]
            aa = a[idx]
            bb = b[idx]
            if aa.size < 2 or bb.size < 2:
                continue

            # one-sided unpaired test
            alt = "greater" if higher_is_better else "less"
            stat = mannwhitneyu(aa, bb, alternative=alt)
            p = float(stat.pvalue)
            p = max(p, np.finfo(float).tiny)  # avoids log(0) that prop to fisher
            p_ctx.append(p)
            
        if not p_ctx:
            continue

        # combine p values across OOD contexts using fisher
        _, p_comb = combine_pvalues(p_ctx, method="fisher")
        p_comb = float(p_comb)

        # check direction for stars
        med_a = float(np.median(a))
        med_b = float(np.median(b))
        anchor_better = (med_a > med_b) if higher_is_better else (med_a < med_b)

        # stars
        if (not anchor_better) or (p_comb >= alpha):
            star = ""
        else:
            if p_comb <= 0.0001:
                star = "****"
            elif p_comb <= 0.001:
                star = "***"
            elif p_comb <= 0.01:
                star = "**"
            elif p_comb <= 0.05:
                star = "*"
            else:
                star = ""

        # log pvalues & stars
        p_values.setdefault(competitor, {})[group] = p_comb
        stars.setdefault(competitor, {})[group] = star

    return p_values, stars
    
def compute_pvalues(
    metrics_dict,
    metric_name,
    higher_is_better=None,
    anchor_method='scDisentangle',
    best_method_only=True,
    alpha=0.05,
):
    import numpy as np
    from scipy.stats import wilcoxon

    if higher_is_better is None:
        if metric_name in correlation_metrics:
            higher_is_better = True
        elif metric_name in distance_metrics:
            higher_is_better = False
        else:
            raise ValueError(f'higher_is_better unclear for {metric_name}')
            
    if anchor_method not in metrics_dict:
        raise ValueError(f"Anchor method '{anchor_method}' not found in metrics_dict.")

    anchor_groups = metrics_dict[anchor_method].keys()

    p_values = {}
    stars = {}

    for method, subdict in metrics_dict.items():
        if method == anchor_method:
            continue

        p_values[method] = {}
        stars[method] = {}

        cands = []
        for group in anchor_groups:
        
            
            a = np.asarray(metrics_dict[anchor_method][group], dtype=float)
            b = np.asarray(subdict[group], dtype=float)

            # same length and at least 2 samples
            if a.shape != b.shape or a.size < 2:
                continue
            
            # greater = scDisentangle better
            if higher_is_better:
                diff = a - b   # positive = scDisentangle larger/better
            else:
                diff = b - a   # positive=scDisentangle smaller/better

            # if diff is zero = no superiority
            if np.allclose(diff, 0):
                p = 1.0
            else:
                # One-sided Wilcoxon signed-rank: H1: median(diff) > 0
                try:
                    stat, p = wilcoxon(diff, alternative='greater')
                except ValueError:
                    raise ValueError('Error in wilcoxon computation')
                    # p = 1.0

            p_values[method][group] = float(p)

            # Median direction check (we only want stars when anchor is better)
            median_diff = float(np.median(diff))

            if median_diff <= 0 or p >= alpha:
                # if scDisentangle not better, or no significance
                star = ""
            else:
                # standard star cutoffs
                if p < 0.0001:
                    star = "****"
                elif p < 0.001:
                    star = "***"
                elif p < 0.01:
                    star = "**"
                elif p < 0.05:
                    star = "*"
                else:
                    star = ""

            stars[method][group] = star

    # after computing p_values and stars normally
    if best_method_only:
        out_p, out_s = {}, {}
        for group in metrics_dict[anchor_method].keys():
            competitors = [m for m in metrics_dict.keys() if m != anchor_method]
            competitor = max(competitors, key=lambda m: np.median(metrics_dict[m][group])) if higher_is_better \
                         else min(competitors, key=lambda m: np.median(metrics_dict[m][group]))
    
            out_p.setdefault(competitor, {})[group] = p_values[competitor][group]
            out_s.setdefault(competitor, {})[group] = stars[competitor][group]
        return out_p, out_s
            
    return p_values, stars
    
def get_results(
    folder_paths, 
    metric_name, 
    perts, 
    n_degs_list = ['All', '200', '100', '50', '20', '10']
    ):
    
    for csv_path in list(folder_paths.values()):
        perts = [x for x in perts if os.path.isfile(f'{csv_path}/{x}.csv')]
    
    print(f'Using {len(perts)} perturbation')
    
    metrics = {k: {v: [] for v in n_degs_list} for k in list(folder_paths.keys())}
    
    for method_name in folder_paths.keys():
        path = folder_paths[method_name]
        for pert in perts:
            
            csv_results = pd.read_csv(f'{path}/{pert}.csv', index_col='Metric')
            
            csv_results = csv_results.loc[metric_name]
            
            for n_degs in n_degs_list:
                metric_value = csv_results[n_degs]
                
                if isinstance(metric_value, str):
                    metric_value = float(metric_value)

                if not np.isfinite(metric_value):
                    raise ValueError('Some values are NAN')
    
                if metric_value < 0:
                    metric_value = 0
                    
                metrics[method_name][n_degs].append(metric_value)

    return metrics
    
def to_np(arrs):
    """
    Convert a list/tuple of AnnData-like objects to a tuple of dense NumPy
    arrays taken from their `.X` attribute.
    """
    arrays = []
    for arr in arrs:
        x = arr.X
        if not isinstance(x, np.ndarray):
            x = x.toarray()
        arrays.append(x)
    return tuple(arrays)

def get_centroid_accuracy(
    _pred_stim, 
    adata_org, 
    pert_name
    ):

    adata_org = adata_org[adata_org.obs['cond_harm'] != 'ctrl'].copy()
    # Get mean per perturbation
    adata_mean = sc.get.aggregate(
        adata_org,
        by="cond_harm",
        func="mean"
    )
    true_mat = adata_mean.layers['mean']
    perts = adata_mean.obs_names.tolist()
    
    # Predicted mean for OOD pert
    pred_vec = _pred_stim.X.mean(axis=0)
    
    # Infex of OOD pert
    correct_index = perts.index(pert_name)
    
    pred_vec = np.asarray(pred_vec).reshape(1, -1)
    true_mat = np.asarray(true_mat)
    d2 = np.sum((true_mat - pred_vec) ** 2, axis=1)  # (K,)
    ranks_pos = np.argsort(np.argsort(d2))        # 0=closest
    rank = ranks_pos[correct_index]
    CRA = 1.0 - (rank / (len(d2) - 1))   

    return {
      'All': {'Centroid Accuracy': CRA*100}
    }
    
def get_sc_similarity(
    _pred_stim, 
    _true_ctrl, 
    ):
   
    _pred_stim, _true_ctrl = to_np([_pred_stim, _true_ctrl])
    
    assert _pred_stim.shape == _true_ctrl.shape, "Shape mismatch"
    N = _pred_stim.shape[0]
    assert N >= 2, ""

    # Pairwise euclidean (NxN)
    a2 = (_pred_stim**2).sum(axis=1, keepdims=True)          # (N,1)
    b2 = (_true_ctrl**2).sum(axis=1, keepdims=True).T        # (1,N)
    D2 = a2 + b2 - 2 * (_pred_stim @ _true_ctrl.T)
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)

    # Single-cell preservation metric
    ranks = np.argsort(D, axis=1).argsort(axis=1)[np.arange(N), np.arange(N)]
    rank_scores = 1.0 - ranks / (N - 1)         # in [0,1]
    rank_mean = float(rank_scores.mean())

    return {
      'All': {'Single-cell Identity preservation': rank_mean*100}
    }
       
def get_correlations(
    _pred_stim,
    _true_stim,
    _pred_ctrl,
    _true_ctrl,
    degs_indices,
    _pert_mean=None,
    degs_list=('All', 200, 100, 50, 20, 10),
):
  
    _pred_stim, _true_stim, _pred_ctrl, _true_ctrl = to_np(
        [_pred_stim, _true_stim, _pred_ctrl, _true_ctrl]
    )

    # Gene-wise u/var across cells
    _pred_stim_mean = _pred_stim.mean(axis=0)
    _true_stim_mean = _true_stim.mean(axis=0)
    _pred_ctrl_mean = _pred_ctrl.mean(axis=0)
    _true_ctrl_mean = _true_ctrl.mean(axis=0)

    metrics = {}

    for n_degs in degs_list:
        if n_degs != 'All':
            degs_mask = degs_indices[:n_degs]
        else:
            degs_mask = slice(None)

        # Mask by DEGS
        pred_stim_mean_masked = _pred_stim_mean[degs_mask]
        true_stim_mean_masked = _true_stim_mean[degs_mask]
        pred_ctrl_mean_masked = _pred_ctrl_mean[degs_mask]
        true_ctrl_mean_masked = _true_ctrl_mean[degs_mask]

        # mean Agreement
        r2_mean = r2_score(true_stim_mean_masked, pred_stim_mean_masked)
        pear_mean, _ = pearsonr(true_stim_mean_masked, pred_stim_mean_masked)

        # delta ctrl
        true_delta_mean = true_stim_mean_masked - true_ctrl_mean_masked
        pred_delta_mean = pred_stim_mean_masked - true_ctrl_mean_masked
        r2_mean_delta = r2_score(true_delta_mean, pred_delta_mean)
        pear_mean_delta, _ = pearsonr(true_delta_mean, pred_delta_mean)
        
        metrics[n_degs] = {
            'R2 Mean': r2_mean *100,
            'Pearson Mean': pear_mean*100,
            'R2 Mean Delta': r2_mean_delta*100,
            'Pearson Mean Delta': pear_mean_delta*100,
        }

        if _pert_mean is not None:

            assert _pert_mean[degs_mask].shape == true_stim_mean_masked.shape
            
            r2_mean_delta_ref = r2_score(
                true_stim_mean_masked - _pert_mean[degs_mask],
                pred_stim_mean_masked - _pert_mean[degs_mask]
                )
    
            pearson_mean_delta_ref, _ = pearsonr(
                true_stim_mean_masked - _pert_mean[degs_mask],
                pred_stim_mean_masked - _pert_mean[degs_mask]
                )
            
            metrics[n_degs].update(
                {
                'Perturbed Reference Pearson Delta': pearson_mean_delta_ref*100,
                'Perturbed Reference R2 Delta': r2_mean_delta_ref*100
                }
            )
    inverted_metrics = {}
    for n_degs, values in metrics.items():
        for metric, value in values.items():
            if metric not in inverted_metrics:
                inverted_metrics[metric] = {}
            inverted_metrics[metric][n_degs] = value

    return inverted_metrics

def get_distances(
    _pred_stim,
    _true_stim,
    _pred_ctrl,
    _true_ctrl,
    degs_indices,
    degs_list=('All', 200, 100, 50, 20, 10),
):
    """
    Distribution metrics
    """
    _pred_stim, _true_stim, _pred_ctrl, _true_ctrl = to_np(
        [_pred_stim, _true_stim, _pred_ctrl, _true_ctrl]
    )

    _pred_stim_mean = _pred_stim.mean(axis=0)
    _true_stim_mean = _true_stim.mean(axis=0)
    _pred_ctrl_mean = _pred_ctrl.mean(axis=0)
    _true_ctrl_mean = _true_ctrl.mean(axis=0)

    metrics = {}

    for n_degs in degs_list:
        if n_degs != 'All':
            degs_mask = degs_indices[:n_degs]
        else:
            degs_mask = slice(None)

        pred_stim_masked = _pred_stim[:, degs_mask]
        true_stim_masked = _true_stim[:, degs_mask]
        pred_ctrl_masked = _pred_ctrl[:, degs_mask]
        true_ctrl_masked = _true_ctrl[:, degs_mask]

        pred_stim_mean_masked = _pred_stim_mean[degs_mask]
        true_stim_mean_masked = _true_stim_mean[degs_mask]
        pred_ctrl_mean_masked = _pred_ctrl_mean[degs_mask]
        true_ctrl_mean_masked = _true_ctrl_mean[degs_mask]

        # Mean Wasserstein distance across genes
        wass_pred = np.mean(
            [
                wasserstein_distance(true_stim_masked[:, i], pred_stim_masked[:, i])
                for i in range(true_stim_masked.shape[1])
            ]
        )

        wass_pred_ctrl = np.mean(
            [
                wasserstein_distance(true_ctrl_masked[:, i], pred_stim_masked[:, i])
                for i in range(true_ctrl_masked.shape[1])
            ]
        )

        # Relative Wasserstein distance (smaller than 1 => better than ctrl)
        RWD = wass_pred / wass_pred_ctrl

        RMSE_mean = np.sqrt(
            ((true_stim_mean_masked - pred_stim_mean_masked) ** 2).mean()
        )

        RMSE_baseline = np.sqrt(
            ((true_stim_mean_masked - true_ctrl_mean_masked) **2).mean()
        )
        Normalized_RMSE_mean = RMSE_mean / RMSE_baseline

        true_delta = true_stim_mean_masked - true_ctrl_mean_masked
        pred_delta = pred_stim_mean_masked - true_ctrl_mean_masked

        RMSE_mean_delta = np.sqrt(
            ((true_delta - pred_delta)**2).mean()
        )

        metrics[n_degs] = {
            'Wasserstein': wass_pred,
            'Relative Wasserstein Distance': RWD,
            'RMSE Mean': RMSE_mean,
            'Normalized RMSE Mean': Normalized_RMSE_mean,
            'RMSE Mean Delta': RMSE_mean_delta
        }

    inverted_metrics = {}
    for n_degs, values in metrics.items():
        for metric, value in values.items():
            if metric not in inverted_metrics:
                inverted_metrics[metric] = {}
            inverted_metrics[metric][n_degs] = value

    return inverted_metrics

def get_specificity_score(
    _pred_stim, 
    _true_stim,
    _true_ctrl,
    _true_pert1,
    _true_pert2,
    degs_indices,
    degs_list = ['All', 200, 100, 50, 20, 10]
    ):

    _pred_stim, _true_stim, _true_ctrl, _true_pert1, _true_pert2 = to_np(
        [_pred_stim, _true_stim, _true_ctrl, _true_pert1, _true_pert2]
        )

    _pred_stim_mean, _true_stim_mean, _true_ctrl_mean, _true_pert1_mean, _true_pert2_mean = _pred_stim.mean(axis=0), _true_stim.mean(axis=0), _true_ctrl.mean(axis=0), _true_pert1.mean(axis=0), _true_pert2.mean(axis=0)
    
    metrics = {}
        
    for n_degs in degs_list:
        
        metrics[n_degs] = {}
        
        if n_degs != 'All':
            
            degs_mask = degs_indices[:n_degs]
            
            pred_stim_masked = _pred_stim[:, degs_mask]
            true_stim_masked = _true_stim[:, degs_mask]
            true_ctrl_masked = _true_ctrl[:, degs_mask]
            true_pert1_masked = _true_pert1[:, degs_mask]
            true_pert2_masked = _true_pert2[:, degs_mask]

            pred_stim_mean_masked = _pred_stim_mean[degs_mask]
            true_stim_mean_masked = _true_stim_mean[degs_mask]
            true_ctrl_mean_masked = _true_ctrl_mean[degs_mask]
            true_pert1_mean_masked = _true_pert1_mean[degs_mask]
            true_pert2_mean_masked = _true_pert2_mean[degs_mask]
        else:
            
            pred_stim_masked = _pred_stim
            true_stim_masked = _true_stim
            true_ctrl_masked = _true_ctrl
            true_pert1_masked = _true_pert1
            true_pert2_masked = _true_pert2

            pred_stim_mean_masked = _pred_stim_mean
            true_stim_mean_masked = _true_stim_mean
            true_ctrl_mean_masked = _true_ctrl_mean
            true_pert1_mean_masked = _true_pert1_mean
            true_pert2_mean_masked = _true_pert2_mean

        wass_pred_stim = [wasserstein_distance(true_stim_masked[:, i].flatten(), pred_stim_masked[:, i].flatten()) for i in range(true_stim_masked.shape[1])]
        wass_pred_ctrl = [wasserstein_distance(true_ctrl_masked[:, i].flatten(), pred_stim_masked[:, i].flatten()) for i in range(true_ctrl_masked.shape[1])]
        wass_pred_pert1 = [wasserstein_distance(true_pert1_masked[:, i].flatten(), pred_stim_masked[:, i].flatten()) for i in range(true_pert1_masked.shape[1])]
        wass_pred_pert2 = [wasserstein_distance(true_pert2_masked[:, i].flatten(), pred_stim_masked[:, i].flatten()) for i in range(true_pert2_masked.shape[1])]

        mae_pred_stim = np.abs(true_stim_mean_masked - pred_stim_mean_masked).mean()
        mae_pred_ctrl = np.abs(true_ctrl_mean_masked - pred_stim_mean_masked).mean()
        mae_pred_pert1 = np.abs(true_pert1_mean_masked - pred_stim_mean_masked).mean()
        mae_pred_pert2 = np.abs(true_pert2_mean_masked - pred_stim_mean_masked).mean()

        correct = 0
        total = len(wass_pred_stim)

        for i in range(total):
            distances = {
                'true_stim': wass_pred_stim[i],
                'ctrl': wass_pred_ctrl[i],
                'pert1': wass_pred_pert1[i],
                'pert2': wass_pred_pert2[i]
            }
            # +1 if true_stim has the smallest distance
            if distances['true_stim'] == min(distances.values()):
                correct += 1

        wass_accuracy = correct / total # Combinatorial specificty score

        if mae_pred_stim == min([mae_pred_stim, mae_pred_ctrl, mae_pred_pert1, mae_pred_pert2]):
            mae_accuracy = 1
        else:
            mae_accuracy = 0

        metrics[n_degs] = {
            'Wasserstein Accuracy': wass_accuracy*100,
            'MAE Accuracy': mae_accuracy*100,
        }

    inverted_metrics = {}

    for n_degs, values in metrics.items():
        for metric, value in values.items():
            if metric not in inverted_metrics:
                inverted_metrics[metric] = {}
            inverted_metrics[metric][n_degs] = value
            
    return inverted_metrics