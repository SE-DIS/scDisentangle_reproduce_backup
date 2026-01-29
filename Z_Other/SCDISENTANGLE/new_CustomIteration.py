import torch
import numpy as np 
import scanpy as sc
from icecream import ic
from tqdm import tqdm
from scipy.stats import wasserstein_distance, pearsonr
import scipy.sparse as sp
from sklearn import metrics
from sklearn.metrics import r2_score
from hygeia.metrics.correlations import average_pearson_correlation 
from scdisentangle.train.tools import train_sklearn
import pandas as pd
import anndata
from copy import deepcopy
from scdisentangle.train.tools import *


class CustomIterations:
    """
    - Class With Custom Train and Evaluations 
    - Inherited by Trained

    Should contain a method 
        extra_custom_train: that specifies what happens after a training epoch 
            (Additional training)

        other custom train methods that will be called by the latter.

    Should contain evaluation functions
        Their name should be the same as in config file: i.e.
            evaluations:
              evaluate_classifiers:
                interval: 3
              evaluate_reconstruction:
                interval: 3
              benchmark:
                interval: 18
              evaluate_and_plot:
                interval: 15
        They will all have access to Trainer attributes.
        Feel free to write and modify them as you see fit.
    """

    def __init__(self):
        super(CustomIterations, self).__init__()

    def extra_cutsom_train(self):
        """
        Define what customized training happens besides the main training loop
        """
        pass

    def to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        else:
            return tensor
    
    def predict(self, adata, counterfactual_dict, bs=256):
        # Should make sure is same as rec when counterfactual_dict is emptuu
        self.toggle_eval()
        self.unfreeze_all()
        
        X = adata.X
        data_size = X.shape[0]
        covariates = {}
        
        for cov in self.hparams['data']['label_keys']:
            covariates[cov] = torch.tensor( [self.dataset.reverse_label_mapping[cov][x] for x in adata.obs[cov].tolist()] ).long()
        
        if not isinstance(X, np.ndarray):
            X = torch.tensor(X.toarray()).float()
        else:
            X = torch.tensor(X).float()
        
        for _cov in self.inp_means.keys():
            self.inp_means[_cov]['latent'] = self.get_latent(
                x_inp=self.inp_means[_cov]['means'],
                )

        preds = {
            'dis_latent_stack': [],
            'cat_latent_stack': [],
            'cat_latent_stack_collapse': [],
            'map_latent_summed_collapse': [],
            'map_latent_summed': [],
            'reconstructed_collapse': []
        }
        
        for batch_idx in tqdm(range(0, data_size, bs)):
            with torch.no_grad():
                X_batch = X[batch_idx:batch_idx+bs].to(self.device)
                covariates_batch = {k:v[batch_idx:batch_idx+bs].to(self.device) for k,v in covariates.items()}
                library = torch.log(X_batch.sum(1)).unsqueeze(1)
                
                pre_latent = self.get_latent(
                    x_inp=X_batch,   
                    )
                
                p_out = self.get_cat_latent(
                    x_inp=pre_latent,
                    variables=covariates_batch,
                    suffixe=''
                    )
        
                counterfactual_latent = self.get_counterfactuals(
                        x_inp=pre_latent,
                        variables=covariates_batch,
                        dis_latent=p_out['dis_latent'],
                        counterfactual_dict=counterfactual_dict,
                        suffixe='_collapse',
                        )
                
                counterfactual_recs = self.get_recs(
                        decoder_name='decoder',
                        decoder_input= counterfactual_latent['map_latent_summed_collapse'],
                        px_name='px_r',
                        library=library,
                        suffixe='_collapse',
                            )
                
                batch_preds = {
                    'dis_latent_stack': p_out['dis_latent_stack'],
                    'cat_latent_stack': p_out['cat_latent_stack'],
                    'cat_latent_stack_collapse': counterfactual_latent['cat_latent_stack_collapse'],

                    'map_latent_summed_collapse': counterfactual_latent['map_latent_summed_collapse'],
                    'map_latent_summed': p_out['map_latent_summed'],
                    'reconstructed_collapse': counterfactual_recs['reconstructed_collapse'],
                }
           
                [preds[k].append(v.cpu().detach()) for k, v in batch_preds.items()]
        
        for key in preds.keys():
            adata.obsm[key] = torch.cat(preds[key], dim=0).numpy()
        
        adata.X = adata.obsm['reconstructed_collapse'].copy()
        del adata.obsm['reconstructed_collapse']

        for key in counterfactual_dict.keys():
           
            adata.obs[f'{key}_org'] = adata.obs[key].copy()
            adata.obs[f'{key}_pred'] = counterfactual_dict[key]
            adata.obs[f'{key}_pred'] = adata.obs[f'{key}_pred'].astype('category')
            adata.obs[key] = adata.obs[f'{key}_pred'].copy()

        return adata

    def get_outputs(self, **kwargs):
        self.toggle_eval()
        self.unfreeze_all()

        outputs = {'data': [], 'indices': []}

        if kwargs['dataloader'] == 'train':
            dataloader = self.train_dataloader
        elif kwargs['dataloader'] == 'test':
            dataloader = self.test_dataloader
        elif kwargs['dataloader'] == 'val':
            dataloader = self.val_dataloader
        elif kwargs['dataloader'] == 'all':
            dataloader = self.dataloader
        elif kwargs['dataloader'] == 'ctrl':
            dataloader = self.ctrl_dataloader
        elif kwargs['dataloader'] == 'null':
            return {}

        else:
            raise ValueError('Dataloader not specified to get_outputs()')
        
        original_values = {}
        if 'collapse' in kwargs.keys():
            collapsed=False
            for _mapper_name in list(kwargs['collapse'].keys()):

                if kwargs['collapse'][_mapper_name]['apply']:
                    original_values[_mapper_name] = self.hparams['growing_neurons']['prior_mappers']['mappers'][_mapper_name]['collapse_name']
                    _collapse_name = kwargs['collapse'][_mapper_name]['collapse_name']

                    ic('Collapsing to', kwargs['collapse'][_mapper_name]['collapse_name'])
                    self.hparams['growing_neurons']['prior_mappers']['mappers'][_mapper_name]['collapse_name'] = _collapse_name
                    collapsed=True 

        with torch.no_grad():
            for indices, data in dataloader:
               
                data = data.to(self.device)

                variables = {}
                for add_var in self.hparams['data']['label_keys']:
                    _tar_var = self.dataset.get_labels_from_ids(indices, add_var).to(self.device)
                    variables[add_var] = _tar_var

                forward_outs = self.forward(
                    x_inp=data, 
                    variables=variables,
                    train_iteration=False,
                    )  

                outputs['data'].append(data.cpu().detach().numpy())
                outputs['indices'].append(indices.cpu().detach().numpy())
                for key in forward_outs.keys():
                    if key not in outputs.keys():
                        outputs[key] = [self.to_numpy(forward_outs[key])] # Potential issue if np
                    else:
                        outputs[key].append(self.to_numpy(forward_outs[key]))

        for out_key in outputs.keys():
            try:
                outputs[out_key] = np.concatenate(outputs[out_key], axis=0)
            except:
                pass
        
        if 'collapse' in kwargs.keys():
            if collapsed:
                for _mapper_name in original_values.keys():
                    self.hparams['growing_neurons']['prior_mappers']['mappers'][_mapper_name]['collapse_name'] = original_values[_mapper_name]

        return outputs

    def get_mig(self, outputs, **kwargs):
        
        metrics = {}

        latent_key = kwargs['latent_key']
        label_keys = kwargs['label_keys']

        latents = outputs[latent_key]

        for label_key in label_keys:
            labels = outputs[label_key]

            Z = np.concatenate((latents, labels.reshape(-1, 1)), axis=1)
            mig_binned = compute_mig(Z, labels.squeeze())

            metrics.update({
                f'MIG_BINNED/{latent_key}_{label_key}': mig_binned
            })

        return metrics

    def evaluate_reconstruction(self, outputs, **kwargs):
        
        rec = outputs[kwargs['rec_key']]
        gt = outputs[kwargs['gt_key']]

        pearson_sample, pearson_gene = average_pearson_correlation(
                rec, gt, mask_zeros = False
                )
        
        mean_r2 = r2_score(gt.mean(axis=0), rec.mean(axis=0))
        variance_r2 = r2_score(gt.var(axis=0), rec.var(axis=0))

        metrics = {
            f'Rec/Pearson_sample': pearson_sample,
            f'Rec/Pearson_gene': pearson_gene,
            f'Rec/R2_mean': mean_r2,
            f'Rec/R2_var': variance_r2,
            }

        return metrics

    def context_transfer_criterion(self, outputs, **kwargs):
  
        metrics = {}
        pred = outputs[kwargs['rec_key']]
        gt = outputs[kwargs['gt_key']]
        stim_name = kwargs['stim_name']
        if isinstance(kwargs['ood_covariate'], str):
            kwargs['ood_covariate'] = [kwargs['ood_covariate']]
        obs_cols = {}
        
        
        _sums = self.dataset.train_anndata.X.sum(axis=1, keepdims=True)
        data_median = np.median(_sums)
        
        if kwargs['normalize_log']:
            pred = normalize_log(pred, target_sum=data_median, log=True)
            gt = normalize_log(gt, target_sum=data_median, log=True)

        for _var in kwargs['variables_of_interest']:
            obs_cols[_var] = [self.dataset.label_mapping[_var][x] for x in outputs[_var]]

        pred_adata = sc.AnnData(X=pred, obs=obs_cols)
        gt_adata = sc.AnnData(X=gt, obs=obs_cols)

        pred_subset = pred_adata
        gt_subset = gt_adata
        ctrl_subset = gt_adata

        ic(pred_subset.shape, gt_subset.shape, ctrl_subset.shape)
        for key, value in kwargs['subsample']['pred'].items():
            pred_subset = pred_subset[pred_subset.obs[key] == value].copy()

        for key, value in kwargs['subsample']['gt'].items():
            gt_subset = gt_subset[gt_subset.obs[key] == value].copy()
        
        for key, value in kwargs['subsample']['ctrl'].items():
            ctrl_subset = ctrl_subset[ctrl_subset.obs[key] == value].copy()
        
        metrics_criterion = {}
        covariate_name = kwargs['eval_by']

        if kwargs['subset_covars']:
            unique_covariates = kwargs['subset_covars']
            unique_covariates = [self.dataset.reverse_label_mapping[covariate_name][x] for x in unique_covariates]
        else:
            unique_covariates = np.unique(outputs[covariate_name])
        
        for cov_name in unique_covariates:

            cov_str = self.dataset.label_mapping[covariate_name][cov_name]
   
            try:
                degs = self.dataset.data.uns[kwargs['adata_degs_key']][stim_name].get(cov_str, [])

                degs_indices = self.dataset.data.var_names.get_indexer(degs)
                degs_indices = degs_indices[degs_indices != -1]  
            except:
                ic('Could not find degs of', stim_name, cov_str)
                continue
            
            for n_degs in kwargs['n_degs_list']:
                degs_mask = degs_indices[:n_degs]
                
                _gt = gt_subset[gt_subset.obs[covariate_name] == cov_str].X[:, degs_mask]
                _pred = pred_subset[pred_subset.obs[covariate_name] == cov_str].X[:, degs_mask]
                _ctrl = ctrl_subset[ctrl_subset.obs[covariate_name] == cov_str].X[:, degs_mask]
                
                computed_metrics = {}

                try:
                    
                    r2_mean = r2_score(_gt.mean(axis=0), _pred.mean(axis=0))
                    if r2_mean < 0:
                        r2_mean = 0

                    computed_metrics['r2_mean'] = r2_mean

                    pear_delta, _ = pearsonr(
                        _gt.mean(axis=0) - _ctrl.mean(axis=0),
                        _pred.mean(axis=0) - _ctrl.mean(axis=0)
                        )
                    if pear_delta < 0:
                        pear_delta = 0

                        computed_metrics['pear_delta'] = pear_delta


                    for metric_name, value in computed_metrics.items():
                        
                        if cov_str in kwargs['ood_covariate']:
                            if len(kwargs['ood_covariate']) > 1:
                                metrics[f'{metric_name}_OOD_{cov_str}/{n_degs}'] = value
                            else:
                                metrics[f'{metric_name}_OOD/{n_degs}'] = value

                        else:
                            key = f'{metric_name}_criterion/{n_degs}'
                            if not key in metrics_criterion:
                                metrics_criterion[key] = []
                            metrics_criterion[key].append(value)
                except Exception as Err:
                    ic('Could not compute metrics for', cov_str)

        metrics_criterion = {k: np.mean(v) if v else 0 for k, v in metrics_criterion.items()}

        # Average criterions
        r2_mean_criterion = {k:v for k,v in metrics_criterion.items() if 'r2_mean_criterion' in k}
        metrics_criterion ['r2_mean_criterion/avg']: np.mean(list(r2_mean_criterion.values()))
        
        metrics.update(metrics_criterion)

        return metrics

    def context_transfer_crispr(self, outputs, **kwargs):
        metrics = {}
        
        label_key = kwargs['label_key']
        ctrl_name = kwargs['ctrl_name']
        rec_key = kwargs['rec_key']

        n_degs_list_ood = kwargs['n_degs_list_ood']
        n_degs_list_val = kwargs['n_degs_list_val']

        ctrl_index = self.dataset.reverse_label_mapping[label_key][ctrl_name]
        ood_labels = self.hparams['OOD']['filter_dict']['filter_single']['labels']
        val_labels = np.unique(self.dataset.data[self.dataset.data.obs['split_ood'] == 'val'].obs[label_key]).tolist()

        gt = outputs["x_inp"]
        gt = normalize_log(gt, target_sum=self.dataset.data.uns['single_perts_median'], log=True)

        labels = outputs[label_key]

        target_labels = ood_labels + val_labels

        metrics_criterion = {}
        for tar_label in target_labels:

            stim_index = self.dataset.reverse_label_mapping[label_key][tar_label]
            ood1, ood2 = tar_label.split('+')
            self.hparams['growing_neurons']['prior_mappers']['mappers']['perturbation1_mapper']['collapse_name'] = ood1
            self.hparams['growing_neurons']['prior_mappers']['mappers']['perturbation2_mapper']['collapse_name'] = ood2

            ctrl_outputs = self.get_outputs(dataloader='ctrl')
            ctrl_labels = ctrl_outputs[label_key]

            rec = ctrl_outputs[rec_key]
            rec = rec[(ctrl_labels==ctrl_index)] # (Should be the same as rec)
            rec = normalize_log(rec, target_sum=self.dataset.data.uns['single_perts_median'], log=True)

            gt_subset = gt[(labels==stim_index)]
            ctrl_subset = gt[(labels==ctrl_index)]

            if tar_label in ood_labels:
                degs_types = ['', '_combo_specific']
                n_degs_list = n_degs_list_ood
            else:
                degs_types = ['']
                n_degs_list = n_degs_list_val

            for degs_type in degs_types:
                _degs = self.dataset.data.uns[f'rank_genes_groups{degs_type}'][tar_label]
                _degs_indices = [self.dataset.data.var_names.get_loc(x) for x in _degs]
                
                for n_degs in n_degs_list:
                    degs_mask = _degs_indices[:n_degs]

                    predicted_mean = rec.mean(axis=0)[degs_mask]
                    gt_mean = gt_subset.mean(axis=0)[degs_mask]
                    ctrl_mean = ctrl_subset.mean(axis=0)[degs_mask]

                    R2_mean = r2_score(gt_mean, predicted_mean)
                    pear_delta, _ = pearsonr(
                        gt_mean - ctrl_mean,
                        predicted_mean - ctrl_mean
                        )

                    if R2_mean < 0:
                        R2_mean = 0
                
                    if tar_label in ood_labels:
                        metrics.update({f'{tar_label}/R2{degs_type}_{n_degs}': R2_mean})
                        metrics.update({f'{tar_label}/Pearson Delta{degs_type}_{n_degs}': pear_delta})
                    else:
                        key = f'r2_mean_criterion/{n_degs}'
                        if not key in metrics_criterion:
                            metrics_criterion[key] = []
                        metrics_criterion[key].append(R2_mean)
                
                    if (n_degs == 10) & (tar_label in ood_labels):
                        wass_distances = [wasserstein_distance(gt_subset[:,degs_mask][:, i], rec[:,degs_mask][:, i]) for i in range(rec[:,degs_mask].shape[1])]
                        wass = np.mean(wass_distances)
                        metrics.update({f'{tar_label}/Wass{degs_type}_{n_degs}': wass})

        metrics_criterion = {k: np.mean(v) if v else 0 for k, v in metrics_criterion.items()}
        r2_mean_criterion = {k:v for k,v in metrics_criterion.items() if 'r2_mean_criterion' in k}
        metrics_criterion['r2_mean_criterion/avg'] = np.mean(list(r2_mean_criterion.values()))
        metrics.update(metrics_criterion)

        return metrics