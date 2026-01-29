import torch
import yaml
from icecream import ic
from hygeia.data.load_rnaseq import DatasetLoader
from scdisentangle.train.trainer import Trainer
from scdisentangle.train.tools import set_seed

if __name__ == '__main__':
    # Read parameters/data/Experiments/Benchmark/Reproducibility/Notebooks/1. PROSTATE/configs/prostate_l2_ood1.yaml
    
    with open('configs/kang.yaml', 'r') as stream:
        hparams = yaml.safe_load(stream)
    
    # Get device
    device = torch.device(
        f'cuda:{hparams["hardware"]["device"]}' if torch.cuda.is_available() else 'cpu'
        )

    if hparams["hardware"]["device"] == 'cpu':
        device = torch.device('cpu')
        
    #device = torch.device('cpu')
    _seed = hparams['train']['set_seed']
    hparams['wandb']['name'] += f' (seed={_seed})'

    set_seed(_seed)

    dataset = DatasetLoader(
        path=hparams['data']['file_path'],
        label_keys=hparams['data']['label_keys'],
        default_normalization=hparams['data']['default_normalization'],
        min_gene_counts=hparams['data']['min_gene_counts'],
        min_cell_counts=hparams['data']['min_cell_counts'],
        n_highly_variable_genes=hparams['data']['highly_variable'],
        use_counts = hparams['data']['use_counts'],
        subset=hparams['data']['SUBSET'],
        seed=_seed,
        )

    dataloader = dataset.get_dataloader(batch_size=2000)

    # dataloader,_,_ = dataset.train_val_test(
    #     test_size=0,
    #     val_size=0,
    #     batch_size=hparams['data']['batch_size']
    #         )

    if hparams['OOD']['apply']:
        train_dataloader, val_dataloader, test_dataloader = dataset.train_val_ood(
            train_size=hparams['data']['train_size'],
            batch_size=hparams['data']['batch_size'],
            filter_dict=hparams['OOD']['filter_dict']
            )
        # train_dataloader, val_dataloader, test_dataloader = dataset.train_val_ood_key(
        #         batch_size=hparams['data']['batch_size'],
        #         split_key=f'split_stimulated_B'
        #      )
    else:
        train_dataloader, val_dataloader, test_dataloader = dataset.train_val_test(
            test_size=hparams['data']['test_size'],
            val_size=hparams['data']['val_size'],
            batch_size=hparams['data']['batch_size'],
            )
    
    ic(len(train_dataloader.dataset), len(val_dataloader.dataset), len(test_dataloader.dataset))
    
    trainer = Trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        hparams=hparams,
        )   

    trainer.train(hparams['train']['main_train']['epochs'])