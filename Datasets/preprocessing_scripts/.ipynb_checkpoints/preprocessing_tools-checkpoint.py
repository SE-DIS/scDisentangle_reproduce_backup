import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def compute_degs(
    adata, 
    cov_key, 
    cond_key, 
    stim_name, 
    control_name, 
    condition_names, 
    synergy=False, 
    method='t-test', 
    rankby_abs=True
    ):
    
    degs_dict = {}
    cell_types = np.append(adata.obs[cov_key].unique(), 'all')
    
    adata_copy = adata.copy()
       
    for cell_type in cell_types:
        try:
            if cell_type == 'all':
                adata_subset = adata_copy[adata_copy.obs[cond_key].isin(condition_names)].copy()
            else:
                adata_subset = adata_copy[(adata_copy.obs[cov_key] == cell_type) & 
                                          (adata_copy.obs[cond_key].isin(condition_names))].copy()
    
            if synergy:
                reference='rest'
            else:
                reference = control_name
            # Perform differential expression analysis
            
            sc.tl.rank_genes_groups(
                adata_subset, 
                cond_key, 
                reference=reference, 
                method=method, 
                use_raw=False, 
                rankby_abs=rankby_abs
            )
            
            # Extract results
            sorted_results = sc.get.rank_genes_groups_df(adata_subset, group=stim_name)
            
            degs_dict[cell_type] = np.array(sorted_results['names'])
        except Exception as Err:
            print('ERROR in ', cell_type, 'SKIPPING..', Err)
    
    return degs_dict

def create_split_cols(
    adata, 
    cov_key, 
    cond_key, 
    stim_name, 
    random_state=42
    ):
    
    np.random.seed(random_state)
    
    for covariate in np.unique(adata.obs[cov_key]):
        # Create masks
        ood_mask = (adata.obs[cov_key] == covariate) & (adata.obs[cond_key] == stim_name)
        non_ood_mask = ~ood_mask
        
        # Initialize split column
        split_col = pd.Series('train', index=adata.obs.index)
        split_col[ood_mask] = 'ood'
        
        # Split non-OOD samples into train/val
        non_ood_idx = adata.obs[non_ood_mask].index
        if len(non_ood_idx) > 0:
            # Stratified split preserves class balance
            train_idx, val_idx = train_test_split(
                non_ood_idx,
                test_size=0.1,
                random_state=random_state,
                stratify=adata.obs[cond_key][non_ood_idx]
            )
            split_col[val_idx] = 'val'
        
        # Add column to adata
        split_col_name = f"split_{stim_name}_{covariate}"
        adata.obs[split_col_name] = split_col.astype('category')
        
    return adata