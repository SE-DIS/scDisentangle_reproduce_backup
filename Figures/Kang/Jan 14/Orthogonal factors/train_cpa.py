import cpa
import scanpy as sc


data_path = '../../../Datasets/preprocessed_datasets/kang.h5ad'

# Read data
adata = sc.read(data_path)

# Set to count
adata.X = adata.layers['counts'].copy()

# Add dose col
adata.obs['dose'] = adata.obs['condition'].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))

# Setup anndata
cpa.CPA.setup_anndata(adata, 
                      perturbation_key='condition',
                      control_group='ctrl',
                      dosage_key='dose',
                      categorical_covariate_keys=[],
                      is_count_data=True,
                      max_comb_len=1,
                     )

# Paramers (set latent to 16 for comparable results)
model_params = {
    "n_latent": 16,
    "recon_loss": "nb",
    "doser_type": "linear",
    "n_hidden_encoder": 128,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 512,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": True,
    "dropout_rate_encoder": 0.0,
    "dropout_rate_decoder": 0.1,
    "variational": False,
    "seed": 6977,
}

trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 50,
    "n_epochs_mixup_warmup": 0,
    "mixup_alpha": 0.0,
    "adv_steps": None,
    "n_hidden_adv": 64,
    "n_layers_adv": 3,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "reg_adv": 20.0,
    "pen_adv": 5.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": True,
    "gradient_clip_value": 1.0,
    "step_size_lr": 10,
}

# Init model
model = cpa.CPA(adata=adata,
                **model_params,
               )

# Train
model.train(max_epochs=2000,
            use_gpu=True,
            batch_size=512,
            plan_kwargs=trainer_params,
            early_stopping_patience=5,
            check_val_every_n_epoch=5,
            save_path='weights_cpa',
           )

# Predict using cell_type and batch embeddings (reconstruct the original gene expressions containing batch effect)
output_batch = model.custom_predict(adata=adata,
                   covars_to_add=[],
                   add_batch=False,
                   add_pert=True,
                   batch_size=2048)

ad = output_batch['latent_z_no_pert']

# Save
ad.write_h5ad('CPA_latent.h5ad')