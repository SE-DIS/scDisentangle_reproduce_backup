import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import yaml
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
    save_path=None,
    split_key=None,
    ctrl_name=None,
    cond_key=None,
    get_full_dataloader=True,
    ):

    set_seed(seed_nb)
    
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

    if get_full_dataloader:
        dataloader = dataset.get_dataloader(batch_size=2000)
    else:
        dataloader=None

    if split_key is not None:
       
        train_dataloader, val_dataloader, test_dataloader = dataset.train_val_ood_key(
            batch_size=hparams['data']['batch_size'],
            split_key=split_key
            )
        ctrl_dataloader = None
    else:
        train_dataloader, val_dataloader, test_dataloader = dataset.train_val_ood(
            train_size=hparams['data']['train_size'],
            batch_size=hparams['data']['batch_size'],
            filter_dict=hparams['OOD']['filter_dict']
            )
        ctrl_dataloader=None
    
    ic(len(train_dataloader.dataset), len(val_dataloader.dataset), len(test_dataloader.dataset))

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
    
    if split_key is not None:
        n_train_anndata, n_test_anndata = trainer.dataset.train_anndata, trainer.dataset.test_anndata
        train_mean, val_mean, test_mean = trainer.dataset.train_anndata.X.mean(), trainer.dataset.val_anndata.X.mean(), trainer.dataset.test_anndata.X.mean()
        with open(f'{save_path}dataloaders_size.txt', 'w') as f:
            f.write(f'train/val/test:{train_size}/{val_size}/{test_size}\n')
            f.write(f'train/test anndata:{n_train_anndata}/{n_test_anndata}\n')
            f.write(f'train/val/test mean:{train_mean}/{val_mean}/{test_mean}\n')

    trainer.train(max_epochs)