import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import numpy as np
import scanpy as sc
from scipy import sparse

def combat_with_reference(
    adata,
    batch_key='level',
    ref_batch=15,
    layer_raw='X_raw',
    copy=False,
):
    """
    Run sc.pp.combat on adata and keep `ref_batch` unchanged by restoring its
    original values after correction.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData.
    batch_key : str
        Column in adata.obs that contains batch labels.
    ref_batch : str
        The batch label (value from adata.obs[batch_key]) that will be kept unchanged.
        If None, function raises an error.
    layer_raw : str or None
        Name of the layer to store the original (pre-ComBat) matrix. If None, original
        is not stored (but we still keep an internal copy to restore the reference).
    copy : bool
        If True, return a copy of adata (like AnnData.copy()); otherwise modify in-place.
    
    Returns
    -------
    adata or None
        If copy=True, returns the corrected AnnData object. Otherwise modifies adata in-place and returns None.
    
    Notes
    -----
    - sc.pp.combat modifies adata.X in-place. We save the pre-combat matrix (optionally in a layer),
      run ComBat, then overwrite rows belonging to ref_batch with the original values.
    - Works with sparse and dense matrices. Row assignment for sparse matrices temporarily converts
      to lil format for efficient row-level assignment.
    - This emulates a reference-based ComBat (R's ref_batch), but is an approximation: it restores
      the *exact original* rows for the reference batch after a global ComBat run.
    """
    if ref_batch is None:
        raise ValueError("ref_batch must be provided (the batch label you want to keep unchanged).")
    if batch_key not in adata.obs.columns:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs")
    if ref_batch not in adata.obs[batch_key].unique():
        raise ValueError(f"ref_batch '{ref_batch}' not found in adata.obs['{batch_key}']")

    if copy:
        adata = adata.copy()

    # Save original matrix (either in layer or as a local var)
    X_original = None
    try:
        if layer_raw is not None:
            # store a copy of adata.X into layer_raw
            adata.layers[layer_raw] = adata.X.copy()
            X_original = adata.layers[layer_raw]
        else:
            X_original = adata.X.copy()
    except Exception:
        # fallback: keep local copy
        X_original = adata.X.copy()

    # Run ComBat (in-place)
    sc.pp.combat(adata, key=batch_key)

    # Build boolean mask for reference samples
    ref_mask = adata.obs[batch_key].astype(str) == str(ref_batch)
    if ref_mask.sum() == 0:
        raise ValueError(f"No samples found for reference batch '{ref_batch}' after mask creation.")

    # Helper: row-assign for dense or sparse matrices
    def _row_assign(target_mat, row_idx_mask, source_mat):
        """
        Replace rows in target_mat where row_idx_mask is True with corresponding rows from source_mat.
        Handles sparse/dense.
        """
        # Get row indices to replace
        rows = np.nonzero(row_idx_mask)[0]
        if len(rows) == 0:
            return target_mat  # nothing to do

        # If source is sparse, convert the selected rows to dense (to preserve values)
        src_is_sparse = sparse.issparse(source_mat)
        tgt_is_sparse = sparse.issparse(target_mat)

        # We'll perform row assignment with lil format when sparse (supports row slicing assignment)
        if tgt_is_sparse:
            tgt_lil = target_mat.tocsr().tolil()
            # iterate rows and assign
            if src_is_sparse:
                src_csr = source_mat.tocsr()
                for r in rows:
                    tgt_lil[r, :] = src_csr[r, :].toarray()
            else:
                for r in rows:
                    tgt_lil[r, :] = source_mat[r, :]

            # convert back to original sparse format (csr recommended)
            return tgt_lil.tocsr()
        else:
            # dense numpy array
            if src_is_sparse:
                src_dense = source_mat.toarray()
            else:
                src_dense = source_mat
            target_mat[rows, :] = src_dense[rows, :]
            return target_mat

    # Restore rows for reference batch from X_original into adata.X
    adata.X = _row_assign(adata.X, ref_mask.values, X_original)

    if copy:
        return adata
    else:
        return None
        
def plot_distance_across_levels(levels_wass, pairs, figsize=(10, 6)):
    """
    Plot pairwise distances across levels for specified cell type pairs.
    
    Parameters:
    -----------
    levels_wass : dict
        Dictionary with level as key and distance DataFrame as value
    pairs : list of tuples
        List of cell type pairs to plot, e.g., [('CD4 T', 'CD8 T'), ('CD8 T', 'NK')]
    figsize : tuple
        Figure size (width, height)
    """
    levels = sorted(levels_wass.keys())
    
    plt.figure(figsize=figsize)
    
    for pair in pairs:
        cell_type_1, cell_type_2 = pair
        distances = []
        
        for level in levels:
            dist_df = levels_wass[level]
            # Get distance (works for both orders since matrix is symmetric)
            if cell_type_1 in dist_df.index and cell_type_2 in dist_df.columns:
                dist = dist_df.loc[cell_type_1, cell_type_2]
            elif cell_type_2 in dist_df.index and cell_type_1 in dist_df.columns:
                dist = dist_df.loc[cell_type_2, cell_type_1]
            else:
                dist = np.nan
            distances.append(dist)
        
        # Plot line for this pair
        label = f"{cell_type_1} - {cell_type_2}"
        plt.plot(levels, distances, marker='o', linewidth=2, markersize=6, label=label)
    
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Wasserstein Distance', fontsize=12)
    plt.title('Cell Type Pair Distances Across Levels', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()
    
def compute_distance(adata, use_rep='X', metric='wasserstein', col='cell_type'):
    """
    Compute pairwise distances between groups in adata.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    use_rep : str
        Representation to use ('X' for .X, or key in .obsm like 'X_pca')
    metric : str
        Distance metric: 'wasserstein', 'mse', or 'pearson'
    col : str
        Column in adata.obs to group by
        
    Returns:
    --------
    pd.DataFrame
        Symmetric distance matrix with groups as indices and columns
    """
    unique_covs = np.unique(adata.obs[col])
    n_covs = len(unique_covs)
    
    # Initialize distance matrix
    dist_matrix = np.zeros((n_covs, n_covs))
    
    # Get data representation
    if use_rep == 'X':
        data = adata.X
        if hasattr(data, 'toarray'):  # Handle sparse matrices
            data = data.toarray()
    else:
        data = adata.obsm[use_rep]
    
    # Compute pairwise distances
    for i, cov_1 in enumerate(unique_covs):
        mask_1 = adata.obs[col] == cov_1
        data_1 = data[mask_1]
        
        for j, cov_2 in enumerate(unique_covs):
            if i > j:  # Skip, will fill by symmetry
                continue
                
            mask_2 = adata.obs[col] == cov_2
            data_2 = data[mask_2]
            
            if metric == 'wasserstein':
                # Average Wasserstein distance across features
                distances = []
                for k in range(data.shape[1]):
                    d = wasserstein_distance(data_1[:, k], data_2[:, k])
                    distances.append(d)
                dist = np.mean(distances)
                
            elif metric == 'mse':
                # MSE between mean vectors
                mean_1 = np.mean(data_1, axis=0)
                mean_2 = np.mean(data_2, axis=0)
                dist = mean_squared_error(mean_1, mean_2)
                
            elif metric == 'pearson':
                # 1 - Pearson correlation between mean vectors
                mean_1 = np.mean(data_1, axis=0)
                mean_2 = np.mean(data_2, axis=0)
                corr = np.corrcoef(mean_1, mean_2)[0, 1]
                dist = 1 - corr
                
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Ensure symmetry
    
    # Convert to DataFrame
    dist_df = pd.DataFrame(dist_matrix, index=unique_covs, columns=unique_covs)
    
    return dist_df
    
def downsample_balance_by_cell_type(adata, key='cell_type', seed=0, copy=True):
    """
    Downsample each cell_type to the minimum group size (perfectly balanced).
    Deterministic given the same adata & seed.
    """
    import numpy as np
    ct = adata.obs[key]
    n_min = ct.value_counts().min()
    rng = np.random.default_rng(seed)

    groups = sorted(ct.value_counts().index.tolist(), key=lambda x: str(x))
    chosen = []
    for g in groups:
        idx = np.flatnonzero(ct == g)
        chosen.append(rng.choice(idx, size=n_min, replace=False))

    chosen = np.concatenate(chosen)
    chosen.sort()  # keep original order
    
    return adata[chosen].copy() if copy else adata[chosen]