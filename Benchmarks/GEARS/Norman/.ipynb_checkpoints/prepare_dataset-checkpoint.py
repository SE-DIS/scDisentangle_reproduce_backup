import scanpy as sc
import scipy.sparse

# Read adata
adata = sc.read_h5ad('../../../Datasets/preprocessed_datasets/norman.h5ad')

print('before pp', adata.shape, adata.X.max(), adata.X.min())

# Normalize
sc.pp.normalize_total(adata, target_sum=adata.uns['single_perts_median'])
sc.pp.log1p(adata)

print('after pp', adata.X.max(), adata.X.min())

# Shape obs columns for GEARS
adata.var['gene_name'] = adata.var_names.tolist()
conditions = adata.obs['cond_harm'].tolist()
conditions = [x+'+ctrl' if not '+' in x else x for x in conditions]
conditions = [x.replace('ctrl+ctrl', 'ctrl') for x in conditions]
adata.obs['condition'] = conditions

if not scipy.sparse.issparse(adata.X):
    adata.X = scipy.sparse.csr_matrix(adata.X)
    
# Prepare data
print('Preparing data..')
from gears import PertData
pert_data = PertData('./data')
pert_data.new_data_process(dataset_name = 'norman', adata = adata)