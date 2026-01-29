import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import yaml
import json
import pandas as pd
from icecream import ic
import os
from hygeia.data.load_rnaseq import DatasetLoader
import hygeia.models.mlp_parts as hmlp
import hygeia.utils.config_tools as hconfig

from scdisentangle.train.trainer import Trainer
from scdisentangle.train.CustomIterations import CustomIterations
from scdisentangle.train.tools import set_seed

def make_run(
    hparams,
    max_epochs,
    seed_nb,
    save_path,
    split_dict,
    ):
    
    set_seed(seed_nb)

    hparams['OOD']['filter_dict']['filter_single']['labels'] = split_dict['ood']
    # Get device
    device = torch.device(
        f'cuda:{hparams["hardware"]["device"]}' if torch.cuda.is_available() else 'cpu'
        )
    
    dataset = DatasetLoader(
            path=hparams['data']['file_path'],
            label_keys=hparams['data']['label_keys'],
            default_normalization=hparams['data']['default_normalization'],
            min_gene_counts=hparams['data']['min_gene_counts'],
            min_cell_counts=hparams['data']['min_cell_counts'],
            n_highly_variable_genes=hparams['data']['highly_variable'],
            use_counts = hparams['data']['use_counts'],
            subset=hparams['data']['SUBSET'],
            )
    dataloader = dataset.get_dataloader(batch_size=2000)

    train_mask = dataset.data.obs['condition'].isin(split_dict['train'])
    val_mask = dataset.data.obs['condition'].isin(split_dict['val'])
    ood_mask = dataset.data.obs['condition'].isin(split_dict['ood'])

    dataset.data.obs['split_ood'] = None
    dataset.data.obs['split_ood'][train_mask] = 'train'
    dataset.data.obs['split_ood'][val_mask] = 'val'
    dataset.data.obs['split_ood'][ood_mask] = 'ood'
    
    train_dataloader, val_dataloader, test_dataloader, ctrl_dataloader = dataset.train_val_ood_key(
        batch_size=hparams['data']['batch_size'],
        split_key='split_ood',
        add_ctrl_dataloader=True,
        ctrl_name='ctrl',
        cond_key='condition'
    )
    
    ic(len(train_dataloader.dataset), len(val_dataloader.dataset), len(test_dataloader.dataset))
    ic(len(ctrl_dataloader.dataset))

    train_size, val_size, test_size = len(train_dataloader.dataset),len(val_dataloader.dataset), len(test_dataloader.dataset) 
    
    trainer = Trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        ctrl_dataloader=ctrl_dataloader,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        hparams=hparams,
        )

    ic(trainer.dataset.data.obs['split_ood'].value_counts())
    
    n_train_anndata, n_test_anndata = trainer.dataset.train_anndata, trainer.dataset.test_anndata
    train_mean, val_mean, test_mean = trainer.dataset.train_anndata.X.mean(), trainer.dataset.val_anndata.X.mean(), trainer.dataset.test_anndata.X.mean()
    with open(f'{save_path}dataloaders_size.txt', 'w') as f:
        f.write(f'train/val/test:{train_size}/{val_size}/{test_size}\n')
        f.write(f'train/test anndata:{n_train_anndata}/{n_test_anndata}\n')
        f.write(f'train/val/test mean:{train_mean}/{val_mean}/{test_mean}\n')

    trainer.train(max_epochs)