


def __init__():

	self.
		train/val/test/ctrl(None)_dataloader
		dataloader
		dataset
		hparams/device/kwargs
		CustomLosses = CustomLosses()
		current_epoch = 0
		best_weights = {
				'R2..': 
				{'epoch': 0,
				'value': 0,
				'models': {},
				'criterion': 'max'
				}
				}

	self._reset_parameters()
		self.losses # same as hparams (deepcopy)
		self.evaluations # same as hparams
		self.models['mapper{idx}/encoder{idx}/mapper_{cov_name}_{mapper}'] (& 'append to lr & models of decoder_optimizer')
		del self.models['encoder']; self.models['mapper']
		self.optimizers
			# {'condition_optimizer': {'models': [], 'lr': [], 'optim': AdamW with params of all these models & weight_decay=0.01
	self.decoder_to_scvi()
		self.pxr = PXR()
		del self.model['decoder']
		self.models['px_r'] = self.px_r
		self.models['deocder'] = DecoderSCVI()

		self.optimizers = self._init_optimizers()
		self.optimizers['px_r'] =  
			# {'optim': torch.optim.AdamW(self.models['px_r'].parameters(), weight_decay=0.01)}

	self.inp_means = self.get_means_norman() if hparams['data']['crispr_data'] 
											 else self.get_means()
def _reset_parameters(self):
	self.
		_init_wandb() # Just init wandb
		_init_variables()
			self.save_gradients & self.exp_path (& creates dir)
			self._optimizers = self.hparams['optimizers']
			self.losses = deepcopy(self.hparams['losses'])
			self.evaluations = self.hparams['evaluations']
		_init_pro_models()
			# Create 
				# self.models['encoder0-15']
				# self.models['mapper0-15']
				# self.models['mapper_cov_name_mapper']
				# push them to hparams['optimizers']['decoder_optimizer']['models']
				# and push [growing_neurons]['lr'] to hparams['optimizers']['lr']
				# Delete encoder & mapper from hparams['models']

				self.models = hconfig.load_models(self.hparams, self.device)

		_create_experiment()
			self.experiment_path # (& Creates directory)

		self.optimizers = self._init_optimizers()
			# {'condition_optimizer': {'models': [], 'lr': [], 'optim': AdamW with params of all these models & weight_decay=0.01
		self.criterions = self._init_criterions()
			# {loss_name: the function} if 'apply'






