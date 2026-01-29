import scanpy as sc
import numpy as np
import anndata as ad
import torch
import yaml

from hygeia.data.load_rnaseq import DatasetLoader
from scdisentangle.train.trainer import Trainer

np.random.seed(42)

def get_predictions(
    coarse_time_label,
    status_control_label,
    ood_cov,
    input_cells_label='Control',
    ):
    
    trainer.hparams['growing_neurons']['prior_mappers']['mappers']['status_control_mapper']['collapse_name'] = status_control_label
    trainer.hparams['growing_neurons']['prior_mappers']['mappers']['coarse_time_mapper']['collapse_name'] = coarse_time_label
    trainer.hparams['growing_neurons']['prior_mappers']['mappers']['zone_mapper']['collapse_name'] = ood_cov

    outputs = trainer.get_outputs(dataloader='train')
    
    predictions = outputs['reconstructed_collapse']
    gt = outputs['x_inp']

    # Covariates
    zone = outputs['zone']
    coarse_time = outputs['coarse_time']
    status_control = outputs['status_control']
    
    status_control = [trainer.dataset.label_mapping['status_control'][x] for x in status_control]
    zone = [trainer.dataset.label_mapping['zone'][x] for x in zone]
    coarse_time = [trainer.dataset.label_mapping['coarse_time'][x] for x in coarse_time]

    obs = {
        'zone_org': zone,
        'status_control_org': status_control,
        'coarse_time_org': coarse_time,
    }
    
    adata_pred = sc.AnnData(
        X=predictions, 
        obs=obs,
        var=trainer.dataset.data.var.copy(),
    )
    
    adata_pred.obs['status_control'] = status_control_label
    adata_pred.obs['coarse_time'] = coarse_time_label
    adata_pred.obs['zone'] = zone
    
    n = adata[(adata.obs['zone'] == ood_cov) & (adata.obs['coarse_time'] == coarse_time_label)\
                & (adata.obs['status_control'] == status_control_label)].shape[0]

    mask = ((adata_pred.obs['zone_org'] == ood_cov) & 
        (adata_pred.obs['coarse_time_org'] == 'Control'))

    adata_pred = adata_pred[mask]
    
    n_available = mask.sum()
    
    # Handle cases where n > available samples
    if n > n_available:
        print(f"Sampling with replacement")
        replace = True
    else:
        replace = False
        
    subsample_indices = np.random.choice(list(range(adata_pred.shape[0])), size=n, replace=replace)
    
    return adata_pred[subsample_indices]
    
for ood_cov in ['Periportal', 'Pericentral']:
    stim_name = 'Infected'
    split_key = f'split_{stim_name}_{ood_cov}'

    # Read yaml
    yaml_path = '../../Benchmarks/SCDISENTANGLE/Liver/configs/liver_hpi.yaml'

    with open(yaml_path, 'r') as stream:
        hparams = yaml.safe_load(stream)
    
    hparams['wandb']['wandb_log'] = False
    hparams['OOD']['filter_dict']['zone'] = ood_cov

    # Load dataloaders
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
    
    # Dataloader cont containing all data samples
    dataloader = dataset.get_dataloader(batch_size=2000)
    
    # Train test val split
    train_dataloader, val_dataloader, test_dataloader = dataset.train_val_ood_key(
            batch_size=hparams['data']['batch_size'],
            split_key=split_key
            )

    # Load trainer
    trainer = Trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        dataloader=dataloader,
        dataset=dataset,
        device=hparams['hardware']['device'],
        hparams=hparams,
    )

    # Load weights
    for model_name in trainer.models.keys():
        weights = torch.load(
            f'../../Benchmarks/SCDISENTANGLE/Liver/weights/liver_hpi/42/{ood_cov}/r2_mean_criterion_10_all/{model_name}.pt'
        )
        trainer.models[model_name].load_state_dict(weights)
        trainer.models[model_name].eval()

    adata = trainer.dataset.data.copy()
    coarse_times = np.unique(adata.obs['coarse_time']).tolist()

    adata_predicted_infected = None
    adata_predicted_uninfected = None
    adata_predicted_ctrl = None

    for coarse_time_label in tqdm(coarse_times):    
        if coarse_time_label == 'Control':
            status_control_label = 'Control'
        else:
            status_control_label = 'Infected'
        
        if coarse_time_label != 'Control':
            adata_pred_infected = get_predictions(
                coarse_time_label=coarse_time_label,
                status_control_label='Infected',
                ood_cov=ood_cov
            )

            adata_pred_uninfected = get_predictions(
                coarse_time_label=coarse_time_label,
                status_control_label='Uninfected',
                ood_cov=ood_cov
            )

            if adata_predicted_infected is None:
                adata_predicted_infected = adata_pred_infected.copy()
                adata_predicted_uninfected = adata_pred_uninfected.copy()
            else:
                adata_predicted_infected = ad.concat([adata_predicted_infected, adata_pred_infected])
                adata_predicted_uninfected = ad.concat([adata_predicted_uninfected, adata_pred_uninfected])
        else:
            adata_pred_control = get_predictions(
                coarse_time_label='Control',
                status_control_label='Control',
                ood_cov=ood_cov
            )
            print('Writing control', ood_cov)
            adata_pred_control.write_h5ad(f'pred_ctrl_{ood_cov}.h5ad')
        
    adata_predicted_infected.write_h5ad(f'pred_infected_{ood_cov}.h5ad')
    adata_predicted_uninfected.write_h5ad(f'pred_uninfected_{ood_cov}.h5ad')