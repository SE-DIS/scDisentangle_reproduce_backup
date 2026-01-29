import os
import random
from copy import deepcopy
from icecream import ic
import numpy as np
import torch 
import torch.nn as nn
from scvi.nn import DecoderSCVI
from scvi.distributions import NegativeBinomial
from tqdm import tqdm
import wandb

import hygeia.utils.config_tools as hconfig
from scdisentangle.train.CustomIterations import CustomIterations
from scdisentangle.train.BaseUtils import BaseTrainUtils
from scdisentangle.losses.custom_losses import CustomLosses
from scdisentangle.train.tools import get_stacked_inputs, get_summed_inputs
from scdisentangle.custom_models.px_r import PXR

class Trainer(BaseTrainUtils, CustomIterations):
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        dataloader,
        dataset,
        device,
        hparams,
        ctrl_dataloader=None,
        **kwargs,
        
        ):
        """
        Main Trainer class

        Parameters
        --------------
        train_dataloader: Pytorch DataLoader
        val_dataloader: Pytorch DataLoader
        test_dataloader: PyTorch DataLoader
        dataloader: PyTorch DataLoader
        dataset: scanpy dataset
        device: PyTorch device
        hparams: Dict
            Config parameters as returned by yaml
        """
        super(Trainer, self).__init__()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.ctrl_dataloader = ctrl_dataloader
        self.dataloader = dataloader
        self.dataset = dataset

        self.hparams = hparams
        self.device = device
        self.kwargs = kwargs    

        self.CustomLosses = CustomLosses(
            device = self.device, 
            )

        self.current_epoch = 0

        criterions = self.hparams['train']['criterions']
        self.best_weights = {k:{'epoch':0, 'value':0, 'models': {} , \
            'criterion': criterions[k]['criterion']} for k in list(criterions.keys())}
        
        self._reset_parameters()

        

    def get_means(self):
        means_dict = {}
        _covariates = [self.hparams['growing_neurons']['prior_mappers']['mappers'][k]['name']\
         for k in self.hparams['growing_neurons']['prior_mappers']['mappers'].keys()]
        
        for _cov in _covariates:

            _adata = self.dataset.data.copy()
 
            _adata.obs['codes'] = _adata.obs[_cov].values.codes.tolist()
            _adata.obs['codes'] = _adata.obs['codes'].astype('category')
            _unique_classes = np.unique(_adata.obs['codes'])
            _means = np.zeros((len(_unique_classes), _adata.shape[1]))
            for i, _cls in enumerate(_unique_classes):
                
                pert_label = self.dataset.label_mapping[_cov][_cls]

                _mean_pert = self.dataset.train_anndata[self.dataset.train_anndata.obs[_cov] == pert_label].X.mean(axis=0)
                _means[i] = _mean_pert

                if np.isnan(_means[i].mean()):
                    raise ValueError(f'Error in computing mean of {_cls} in {_cov}')

            _means = torch.tensor(_means).to(self.device).float()

            means_dict[_cov] = {'means': _means, 'codes': _unique_classes.tolist()}

        return means_dict

    def get_means_norman(self):
        means_dict = {}
        _covariates = [self.hparams['growing_neurons']['prior_mappers']['mappers'][k]['name']\
         for k in self.hparams['growing_neurons']['prior_mappers']['mappers'].keys()]
        
        for _cov in _covariates:

            _adata = self.dataset.data.copy()
           
            _adata.obs['codes'] = _adata.obs[_cov].values.codes.tolist()
            _adata.obs['codes'] = _adata.obs['codes'].astype('category')
            _unique_classes = np.unique(_adata.obs['codes'])
            _means = np.zeros((len(_unique_classes), _adata.shape[1]))
            for i, _cls in enumerate(_unique_classes):
                
                pert_label = self.dataset.label_mapping[_cov][_cls]
                unique_perts = np.unique(self.dataset.train_anndata.obs['condition']) 
                #subset_perts = [x for x in unique_perts if x.startswith(pert_label + '+') or x.endswith('+' + pert_label)]
                subset_perts = [x for x in unique_perts if pert_label in x]
                if pert_label == 'NOPERT':
                    _mean_pert = self.dataset.train_anndata[self.dataset.train_anndata.obs['condition'] == 'ctrl'].X.mean(axis=0)
                else:
                    _mean_pert = self.dataset.train_anndata[self.dataset.train_anndata.obs['condition'].isin(subset_perts)].X.mean(axis=0)
                
                _means[i] = _mean_pert

                if np.isnan(_means[i].mean()):
                    ic(pert_label,_cov, _means[i].mean(), len(subset_perts), len(self.dataset.train_anndata),len(self.dataset.test_anndata), len(unique_perts) )
                    ic(self.dataset.val_anndata)
            _means = torch.tensor(_means).to(self.device).float()

            means_dict[_cov] = {'means': _means, 'codes': _unique_classes.tolist()}
        return means_dict

    def decoder_to_scvi(self):

        self.px_r = PXR(self.hparams['models']['decoder']['layers'][-1]).to(self.device)
        del self.models['decoder']

        self.models['px_r'] = self.px_r
        self.models['decoder'] = DecoderSCVI(
            n_input=self.hparams['models']['decoder']['layers'][0],
            n_output=self.hparams['models']['decoder']['layers'][-1],
            n_layers=self.hparams['decoder_parameters']['use_scvi_decoder']['n_layers'],
            n_hidden=self.hparams['decoder_parameters']['use_scvi_decoder']['n_hidden'],
            use_batch_norm=self.hparams['models']['decoder']['batch_norm'],
            use_layer_norm=False,
        ).to(self.device)

        self.optimizers = self._init_optimizers()
        self.optimizers['px_r'] = {'optim': torch.optim.AdamW(self.models['px_r'].parameters(), weight_decay=0.01)}
            
    def _init_wandb(self): 
        
        hconfig.init_wandb(self.hparams)
      
        if self.hparams['wandb']['wandb_log']:
            name = self.hparams['wandb']['name']
            group = self.hparams['wandb']['group']
            project = self.hparams['wandb']['project']
            
    def _init_pro_models(self):

        # Init prior mappers
        total_emb_dim = 0
        for prior_mapper in self.hparams['growing_neurons']['prior_mappers']['mappers'].keys():

            embedder_name = self.hparams['growing_neurons']['prior_mappers']['mappers'][prior_mapper]['embedder_name']
            n_emb = self.hparams['embedders'][embedder_name]['n_dim']
                
            self.hparams['models'][f'mapper_{prior_mapper}'] = deepcopy(self.hparams['models']['mapper'])
            self.hparams['models'][f'mapper_{prior_mapper}']['layers'][0] = total_emb_dim + n_emb #n_emb

            self.hparams['optimizers']['decoder_optimizer']['models'].append(f'mapper_{prior_mapper}')
            self.hparams['optimizers']['decoder_optimizer']['lr'].append(self.hparams['growing_neurons']['lr'])

            total_emb_dim += n_emb       

        # Init encoders & mappers
        for idx in range(self.hparams['growing_neurons']['total_neurons']):
            
            _lr = self.hparams['growing_neurons']['lr']

            # Add encoder
            self.hparams['models'][f'encoder{idx}'] = deepcopy(self.hparams['models']['encoder'])
            self.hparams['optimizers']['decoder_optimizer']['models'].append(f'encoder{idx}')
            self.hparams['optimizers']['decoder_optimizer']['lr'].append(_lr)

            # Add mapper
            self.hparams['models'][f'mapper{idx}'] = deepcopy(self.hparams['models']['mapper'])
            self.hparams['models'][f'mapper{idx}']['layers'][0] += (idx + total_emb_dim)
            self.hparams['optimizers']['decoder_optimizer']['models'].append(f'mapper{idx}')
            self.hparams['optimizers']['decoder_optimizer']['lr'].append(self.hparams['growing_neurons']['lr'])


        # Del template encoder & mapper
        del self.hparams['models']['encoder']
        del self.hparams['models']['mapper'] 

        # Init models
        self.models = hconfig.load_models(
            self.hparams, 
            self.device
            )

    def _reset_parameters(self):
        
        self._init_wandb()
        self._init_pro_models()
        self.create_experiment()

        self.criterions = self._init_criterions()

        if self.hparams['decoder_parameters']['use_scvi_decoder']['apply']:
            self.decoder_to_scvi()
        else:
            self.optimizers = self._init_optimizers()

        if self.hparams['data']['crispr_data']:
            self.inp_means = self.get_means_norman()
        else:
            self.inp_means = self.get_means()
        
    def _init_criterions(self):
        # Init criterions from the specified infos
        criterions = {
            k:getattr(self.CustomLosses, v['fnc_name']) for k,v in self.hparams['losses'].items() if v['apply']
        }

        return criterions

    def _init_optimizers(self):
        # Init optimizers from the specified info in optimizers dict
        optimizers = deepcopy(self.hparams['optimizers'])

        for optimizer_name, optimizer_dict in optimizers.items():
            params = []
            # > Python 3.7 for keys to be ordered
            for idx, model_name in enumerate(optimizer_dict['models']):
                model = self.models[model_name]
                lr = optimizer_dict['lr'][idx]
                params.append({'params': model.parameters(), 'lr': lr})

            optimizers[optimizer_name]['optim'] = torch.optim.AdamW(params, weight_decay=0.01)

        return optimizers
    
    def get_latent(
        self, 
        x_inp, 
        ):
        
        if self.hparams['data']['use_counts']:
            _x_inp = torch.log(1 + x_inp)
        else:
            _x_inp = x_inp

        latent = self.models['encoder_pretrain'](_x_inp)

        return latent

    def get_counterfactuals(
        self,
        x_inp,
        dis_latent,
        variables,
        suffixe='collapse',
        counterfactual_dict=None
        ):

        cat_latent = []
        map_latent = []
        prior_mapers_names = list(self.hparams['growing_neurons']['prior_mappers']['mappers'].keys())

        for mapper_name in prior_mapers_names:
            
            name = self.hparams['growing_neurons']['prior_mappers']['mappers'][mapper_name]['name']
            embedder_name = self.hparams['growing_neurons']['prior_mappers']['mappers'][mapper_name]['embedder_name']
            targets = variables[name]
            
            if counterfactual_dict is not None:
                if mapper_name.replace('_mapper', '') in counterfactual_dict.keys():
                    collapse_target = True
                    collapse_name = counterfactual_dict[mapper_name.replace('_mapper', '')]
                else:
                    collapse_target = False
            else:
                collapse_target = self.hparams['growing_neurons']['prior_mappers']['mappers'][mapper_name]['collapse_target']
                collapse_name = self.hparams['growing_neurons']['prior_mappers']['mappers'][mapper_name]['collapse_name']

            if collapse_target:
                col_idx = self.dataset.reverse_label_mapping[name][collapse_name]
                col_targets = torch.ones(targets.shape, dtype=torch.long).to(self.device) * col_idx
                sub_emb = torch.sin(self.models[embedder_name](col_targets))
            else:
                sub_emb = torch.sin(self.models[embedder_name](targets))
              
            for emb_dimension in range(sub_emb.shape[1]):
                cat_latent.append(sub_emb[:, emb_dimension].unsqueeze(1))
            
            cat_prior_latents = torch.cat(cat_latent, dim=1)
            mapped_latent = self.models[f'mapper_{mapper_name}'](cat_prior_latents)
            zeros = torch.zeros_like(x_inp).to(self.device)
                
            for idx in torch.unique(targets):
                if collapse_target:
                    _code_idx = self.inp_means[name]['codes'].index(col_idx)
                else:
                    _code_idx = self.inp_means[name]['codes'].index(idx)
                # Assert code_idx == idx, also assert zeros has no 0s
                zeros[targets==idx] = self.inp_means[name]['latent'][_code_idx]
            
            map_latent.append(zeros)
            map_latent.append(mapped_latent)

        assert len(dis_latent) == self.hparams['growing_neurons']['total_neurons']
        for idx in range(self.hparams['growing_neurons']['total_neurons']):
            cat_latent.append(dis_latent[idx])
            cat_sub_latents = torch.cat(cat_latent, dim=1)
            mapped_latent = self.models[f'mapper{idx}'](cat_sub_latents)
                
            map_latent.append(mapped_latent)

        cat_latent_stack = get_stacked_inputs(inputs_list=cat_latent)
        map_latent_summed = get_summed_inputs(inputs_list=map_latent)

        outs = {
            'cat_latent': cat_latent,
            'map_latent': map_latent,
            'cat_latent_stack': cat_latent_stack,
            'map_latent_summed': map_latent_summed,
        }

        outs = {k+suffixe:v for k,v in outs.items()}

        return outs

    def get_cat_latent(
        self,
        x_inp,
        variables=None,
        suffixe='',
        ):

        dis_latent=[]
        cat_latent=[]
        map_latent=[]

        prior_mapers_names = list(self.hparams['growing_neurons']['prior_mappers']['mappers'].keys())

        for mapper_name in prior_mapers_names:
            
            name = self.hparams['growing_neurons']['prior_mappers']['mappers'][mapper_name]['name']
            embedder_name = self.hparams['growing_neurons']['prior_mappers']['mappers'][mapper_name]['embedder_name']
            targets = variables[name]
            
            collapse_target = self.hparams['growing_neurons']['prior_mappers']['mappers'][mapper_name]['collapse_target']
            collapse_name = self.hparams['growing_neurons']['prior_mappers']['mappers'][mapper_name]['collapse_name']
              
            sub_emb = torch.sin(self.models[embedder_name](targets)) # COLLAPSE IDX..
            for emb_dimension in range(sub_emb.shape[1]):
                cat_latent.append(sub_emb[:, emb_dimension].unsqueeze(1))
            
            cat_prior_latents = torch.cat(cat_latent, dim=1)
            mapped_latent = self.models[f'mapper_{mapper_name}'](cat_prior_latents)
            zeros = torch.zeros_like(x_inp).to(self.device)
            
            for idx in torch.unique(targets):
                _code_idx = self.inp_means[name]['codes'].index(idx)
                # Assert code_idx == idx, also assert zeros has no 0s
                zeros[targets==idx] = self.inp_means[name]['latent'][_code_idx]
            
            map_latent.append(zeros)
            map_latent.append(mapped_latent)

        for idx in range(self.hparams['growing_neurons']['total_neurons']):
            map_subtraction = map_latent
            _inp = x_inp - get_summed_inputs(inputs_list=map_subtraction)
            sub_latent = self.models[f'encoder{idx}'](_inp)

            dis_latent.append(sub_latent)
            cat_latent.append(sub_latent)
            cat_sub_latents = torch.cat(cat_latent, dim=1)
            mapped_latent = self.models[f'mapper{idx}'](cat_sub_latents)
                
            map_latent.append(mapped_latent)

        cat_latent_stack = get_stacked_inputs(inputs_list=cat_latent)
        dis_latent_stack = get_stacked_inputs(inputs_list=dis_latent)
        map_latent_summed = get_summed_inputs(inputs_list=map_latent)

        outs = {
            'dis_latent': dis_latent,
            'cat_latent': cat_latent,
            'map_latent': map_latent,
            'cat_latent_stack': cat_latent_stack,
            'dis_latent_stack': dis_latent_stack,
            'map_latent_summed': map_latent_summed,
        }
        
        outs = {k+suffixe:v for k,v in outs.items()}

        return outs

    def get_recs(
        self, 
        decoder_name, 
        decoder_input, 
        px_name,
        library, 
        suffixe,
        ):

        if self.hparams['decoder_parameters']['use_scvi_decoder']['apply']:
            
            px_scale, _, px_rate, px_dropout = self.models[decoder_name](
                'gene', 
                decoder_input, 
                library,
                )

            px_r = torch.exp(self.models[px_name].px_r)
            px_l = NegativeBinomial(mu = px_rate, theta = px_r)
            reconstructed = px_rate

            outs = {
                'px_scale': px_scale,
                'px_rate':px_rate,
                'px_dropout':px_dropout,
                'reconstructed':reconstructed,
                'px_l':px_l,
                'px_r': px_r
            }
        else:
            reconstructed = self.models[decoder_name](decoder_input)
            outs = {'reconstructed': reconstructed}

        outs = {k + suffixe:v for k,v in outs.items()}

        return outs

    def forward(
        self, 
        x_inp, 
        variables,
        train_iteration=False,
        ):

        x_inp = x_inp.to(self.device)
        library = torch.log(x_inp.sum(1)).unsqueeze(1)

        forward_outs = {}

        pre_latent = self.get_latent(
            x_inp=x_inp,   
            )

        for _cov in self.inp_means.keys():
            self.inp_means[_cov]['latent'] = self.get_latent(
                x_inp=self.inp_means[_cov]['means'],
                )
        
        p_out = self.get_cat_latent(
            x_inp=pre_latent,
            variables=variables,
            suffixe=''
            )

        forward_outs.update(p_out)

        _recs_pretrain = self.get_recs(
            decoder_name='decoder',
            decoder_input=pre_latent,
            px_name='px_r',
            library=library,
            suffixe='',
            )
        
        forward_outs.update(_recs_pretrain)

        if not train_iteration:
            
            counterfactual_latent = self.get_counterfactuals(
                x_inp=pre_latent,
                variables=variables,
                dis_latent=p_out['dis_latent'],
                suffixe='_collapse',
                )

            forward_outs.update(counterfactual_latent)

            counterfactual_recs = self.get_recs(
                decoder_name='decoder',
                decoder_input= counterfactual_latent['map_latent_summed_collapse'],
                px_name='px_r',
                library=library,
                suffixe='_collapse',
                    )
            forward_outs.update(counterfactual_recs)

        forward_outs.update(variables)
        forward_outs.update(
            {
            'x_inp': x_inp,
            'pre_latent': pre_latent,
            }
            )   

        return forward_outs
    
    def log_epoch(self):
        if self.hparams['wandb']['wandb_log']:
            with open(f'{self.wandb_project_path}/n_epochs.txt', 'a') as f:
                f.write(f'{self.current_epoch}\n')
            
    def train_epoch(self):
        
        self.CustomLosses.current_epoch = self.current_epoch
        self.toggle_train()
        epoch_outs = {}
        
        self.log_epoch()
        for indices, data in self.train_dataloader:
           
            data = data.to(self.device)
            batch_losses = {}

            self.zero_grad_optimizers()
            self.unfreeze_all()

            variables = {}
            for l_key in self.hparams['data']['label_keys']:
                _target_codes = self.dataset.get_labels_from_ids(indices, l_key).to(self.device)
                variables[l_key] = _target_codes

            forward_outs = self.forward(
                x_inp=data, 
                variables=variables,
                train_iteration=True,
                )

            for param_loss in self.hparams['losses'].keys():
                if self.hparams['losses'][param_loss]['apply']:
                    if self.isapply(self.hparams['losses'][param_loss]):
                        batch_losses[param_loss] = self.criterions[param_loss](
                            forward_outs=forward_outs,
                            loss_dict=self.hparams['losses'][param_loss]
                            )

            self.update_losses(
                batch_losses=batch_losses, 
                epoch_outs=epoch_outs
                )

            self.step_all()

        epoch_outs = self.normalize_losses(epoch_outs, len(self.train_dataloader))
        epoch_outs = self.add_prefix('losses/', epoch_outs)

        return epoch_outs

    def evaluate(self, epoch=1):
        
        """
        Evaluations shoud be stated in the config or as a dict
        Along with their interval
        and defined in CustomIterations class
        """
        if self.hparams['train']['evaluate'] is False:
            return {}

        self.toggle_eval()
        eval_metrics = {}

        outputs = {
            'train': None,
            'test': None,
            'val': None,
            'ctrl': None,
            'all': None,
            'null': None
        }

        for evaluation_name, evaluation in self.hparams['evaluations'].items():
            if evaluation['interval']:
                if self.current_epoch % evaluation['interval'] == 0:
                    if not ((epoch == 0) and (evaluation['start_eval'] == False)):
                        if self.current_epoch > evaluation['start_eval'] and self.current_epoch < evaluation['stop_eval']:
            
                            kwargs = evaluation['kwargs']
                            for out in evaluation['dataloaders']:

                                if outputs[out] is None:
                                    outputs[out] = self.get_outputs(
                                        dataloader=out,
                                        **kwargs
                                        )
                                
                            for dataloader_name in evaluation['dataloaders']:
                                metrics = getattr(
                                        self, evaluation_name.lower()
                                        )(outputs=outputs[dataloader_name],**kwargs)
                                metrics = {f'{k}_{dataloader_name}':v for k,v in metrics.items()}
                                
                                eval_metrics.update(metrics)

        return eval_metrics
    
    def train(self, n_epochs=None):
        # Train and eval for n_epochs
        if n_epochs is None:
            n_epochs = self.hparams['train']['main_train']['epochs']
        
        pbar = tqdm(range(n_epochs), desc="Epochs")
        for epoch in pbar:#range(n_epochs):
            self.gradient_by_loss = {}
            # Record metrics
            metrics = {}

            # 1 training step
            losses = self.train_epoch()

            # Additional training step (in CustomIterations)
            self.extra_cutsom_train()

            # Save weights if specified in config
            self.save_weights()

            #self.current_epoch += 1
            eval_metrics = self.evaluate(epoch=epoch)
            # Update metrics
            metrics.update(losses)
            metrics.update(self.gradient_by_loss)
            metrics.update(eval_metrics)
                        
                         
            self.log_best_weights(eval_metrics)
            self.current_epoch += 1
            self.log(metrics)


    def log_best_weights(self, metrics):
        for m in metrics.keys():
            if m in self.best_weights.keys():
                current_best_value = self.best_weights[m]['value']
                criterion = self.best_weights[m]['criterion']
                if criterion == 'max':
                    inequality_criterion = (current_best_value < metrics[m])
                elif criterion == 'min':
                    inequality_criterion = (current_best_value > metrics[m])
                else:
                    raise ValueError(f'criterion of {m} should be either max or min')

                if inequality_criterion or current_best_value == 0:

                    self.best_weights[m]['value'] = metrics[m]
                    self.best_weights[m]['epoch'] = self.current_epoch

                    if self.hparams['save_experiment']['save_best_weights']['apply']:
                        for model_name in self.models.keys():
                           self.best_weights[m]['models'][model_name] = deepcopy(self.models[model_name].state_dict())

                        s_path = self.hparams['save_experiment']['experiment_path'] + '/' + m.replace('/', '_') + '/'
                        self.to_pt(s_path)
                        with open(s_path + 'epoch_nb.txt', 'a') as f:
                            f.write(str(self.current_epoch) + '\n')
                        with open(s_path + 'value.txt', 'a') as f:
                            f.write(str(metrics[m]) + '\n')
                
    def log(self, metrics):
        """
        Log train and eval metrics
        """
        metrics_no_grad = {k:v for k,v in metrics.items() if not k.startswith('Gradients_by_loss')}

        if self.hparams['wandb']['wandb_log']:
            wandb.log(metrics, step=self.current_epoch)

        if self.hparams['other']['verbose']:
            print(metrics_no_grad)
