import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def em_cluster(
    adata,
    n_comps,
    cell_type_key='cell_type',
    obsm_key=None,
    n_pcs=30,
    standardize=True,
    random_state=42,
    store_key='em_cluster',
    covariance_type='full'
    ):
    if obsm_key is not None:
        X = adata.obsm[obsm_key].copy()
    else:
        X = adata.X.copy()

    if standardize:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        
    if n_pcs is not None:
        print('Applying PCA')
        n_components = int(min(n_pcs, X.shape[1]))
        X = PCA(n_components=n_components, random_state=random_state).fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_comps,
        covariance_type=covariance_type,
        init_params='kmeans',
        random_state=random_state,
    )
    gmm.fit(X)
    labels = gmm.predict(X)

    adata.obs[store_key] = pd.Categorical(labels.astype(int))

    cell_types = adata.obs[cell_type_key].astype(str)
    counts = pd.crosstab(cell_types, labels)
    
    return labels

def get_tree(
    adata, 
    final_level=15,
    step=2,
    cell_type_key='cell_type',

    n_comps=15,
    standardize=True,
    random_state=42,
    n_pcs=30,
    obsm_key=None,
    dynamic_comps=False,
    ):

    if step is None:
        n_unique = adata.obs['cell_type'].nunique()
        if n_unique <= 1:
            raise ValueError("Need at least 2 cell types to compute step.")
        
        computed_step = int(16 // (n_unique - 1))
        if computed_step < 1:
            computed_step = 1
        
        step = computed_step
        
    clusters = adata.obs['cell_type'].unique().tolist()
    new_clusters = deepcopy(clusters)
    
    mappings = {}
    merged_levels = {}
    
    current_level = final_level
    iter_step = 1
    rows = []
    while True:
        
        adata_subset = adata[adata.obs['level'] == current_level].copy()
        adata_subset.obs[cell_type_key] = adata_subset.obs[cell_type_key].astype(str)
                
        for mapping_key, mapping_value in mappings.items():
            adata_subset.obs[cell_type_key] = adata_subset.obs[cell_type_key].replace(mapping_key, mapping_value)

        if dynamic_comps is not False:
            n_comps = len(adata_subset.obs[cell_type_key].unique()) + dynamic_comps
    
        labels = em_cluster(
            adata=adata_subset,
            n_comps=n_comps,
            cell_type_key=cell_type_key,
            obsm_key=obsm_key,
            n_pcs=n_pcs,
            standardize=standardize,
            random_state=0,
        )
        
        counts_df = pd.crosstab(adata_subset.obs[cell_type_key], labels)
        counts_df = (counts_df.T / counts_df.sum(axis=1)).T
    
        # rows already normalized to sum=1
        X = counts_df.fillna(0.0).to_numpy()
        G = X @ X.T  # pairwise dot products (similarities)
        np.fill_diagonal(G, -np.inf)  # exclude self
    
        i, j = np.unravel_index(np.argmax(G), G.shape)
        best_pair = (counts_df.index[i], counts_df.index[j])
        best_score = float(G[i, j])
    
        best_pair = list(best_pair)
        for pair in best_pair:
            mappings[pair] = '&'.join(best_pair)
        merged_levels[current_level] = best_pair
        
        current_level -= step

        n_samples = sum([_.count('&') + 1 for _ in best_pair])
        row = [new_clusters.index(best_pair[0]), new_clusters.index(best_pair[1]), float(iter_step), n_samples]
        rows.append(row)
        new_clusters.append('&'.join(best_pair))
        iter_step += 1
        
        if len(counts_df) == 2:
            break
        elif current_level < 0:
            break

    linkage_mtx = np.array(rows)

    tree_results = {
        'merged_levels': merged_levels,
        'linkage_mtx': linkage_mtx,
        'clusters': clusters
    }
    return tree_results

def plot_dendro(
    linkage_mtx, 
    clusters, 
    figsize=(14, 6),
    save_path=None,
    linewidth=2.5,
    orientation='bottom',
    label_fontsize=14,
):
    plt.figure(figsize=figsize)

    # Let dendrogram handle all coordinates & labels
    dendrogram(
        linkage_mtx,
        labels=clusters,
        orientation=orientation,
    )

    ax = plt.gca()

    # Increase line width for all branch collections (this is what dendrogram actually uses)
    for coll in ax.collections:
        coll.set_linewidth(linewidth)

    # Optional: bigger tick/label font, but no change to positions
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)

    if save_path is not None:
        print(f'saving to {save_path}')
        plt.savefig(save_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(save_path + '.pdf', dpi=600, bbox_inches='tight')

    plt.show()