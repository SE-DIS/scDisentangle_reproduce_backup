import torch 
import numpy as np
import os
import hygeia.utils.directory_manager as hdm
from icecream import ic

class BaseTrainUtils:
    """
    Contained base parameters for model training

    Inherited by Trainer, see config file for conventions
    """
    def __init__(self):
        super(BaseTrainUtils, self).__init__()

    def create_experiment(self):
        if 'save_experiment' in self.hparams.keys():
            if self.hparams['save_experiment']['apply']:
                if 'experiment_path' in self.hparams['save_experiment'].keys():
                    self.experiment_path = self.hparams['save_experiment']['experiment_path']
                else:
                    self.experiment_path = self.hparams['wandb']['name']

                self.weights_path = f'{self.experiment_path}/weights/'
                hdm._make_tree(self.weights_path)

    def save_weights(self):
        if self.hparams['save_experiment']['save_weights']['apply']:
            if self.current_epoch % self.hparams['save_experiment']['save_weights']['interval'] == 0:
                current_save_path = f'{self.weights_path}{self.current_epoch}/'
                
                self.to_pt(current_save_path)
    
    def isapply(self, fnc):
        apply_statement = False
        if fnc['activate_at'] <= self.current_epoch:
            if fnc['stop_at'] > self.current_epoch:
                apply_statement = True

        return apply_statement

    def to_pt(self, save_path):
        hdm._make_tree(save_path)
        for model_name, model in self.models.items():
            torch.save(
                model.state_dict(), 
                f'{save_path}{model_name}.pt'
                )

    def push_to_device(self):
        for model_name in self.models.keys():
            self.models[model_name] = self.models[model_name].to(
                self.device
                )

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def toggle_train(self):
        for model_name, model in self.models.items():
            model.train()

    def toggle_eval(self):
        for model_name, model in self.models.items():
            model.eval()

    def freeze_all(self, _except=[]):
        """
        Freeze all models in parent class except 
        specified models, to control gradient flow
        customed to each loss

        Parameters
        --------------
        _except: List
            name of models to keep unfreezed
        """
        for model_name, model in self.models.items():
            if model_name not in _except:
                self.freeze(model)
            

    def unfreeze_all(self, _except=[]):
        """
        Unfreeze all models in parent class except 
        specified models

        Parameters
        --------------
        _except: List
            name of models to keep freezed
        """
        for model_name, model in self.models.items():
            if model_name not in _except:
                self.unfreeze(model)

    def zero_grad_optimizers(self):
        for optimizer_name, optimizer in self.optimizers.items():
            optimizer['optim'].zero_grad()

    def step_all(self):
        for optimizer_name, optimizer in self.optimizers.items():
            optimizer['optim'].step()

    def add_loss(self, loss_name, loss_dict, computed_loss):
        """
        add a loss to an existing dictionary of losses

        Parameters
        --------------
        loss_name: Str 
            Name of the loss as in the config
        loss_dict:
            Dictionary containing loss_name: value
        computed_loss: Float/Tensor
            current loss to add to existing losses dict
        """
        if isinstance(computed_loss, torch.Tensor):
            computed_loss = computed_loss.item()

        if loss_name in loss_dict:
            loss_dict[loss_name] += computed_loss
        else:
            loss_dict[loss_name] = computed_loss
        
    def normalize_losses(self, loss_dict, total):
        """
        Divide all losses in a dict by the total (len(dataloader))
        """
        return {k:v/total for k,v in loss_dict.items()}

    def get_weighted_loss(self, computed_loss, loss_name):
        """
        Return weighted loss as specified in the config and parent class

        if it's a string pointing to another loss, an additional key of ratio
        is expected so that the weight is adaptive on another loss

        Parameters
        --------------
        computed_loss: Float/Tensor
            Current computed loss
        loss_name: Name of the loss as specified in the config / parent class

        Returns
        --------------
        computer_loss: torch.Tensor/float
            The weighted loss
        """
        weight = self.losses[loss_name]['weight']
        if (isinstance(weight, int)) or (isinstance(weight, float)):
            computed_loss = computed_loss * weight
        else:
            raise ValueError(f'Error in {loss_name} weight: missing')

        return computed_loss

    def update_losses(self, batch_losses, epoch_outs):
        """
        Process a computed loss:
            backward to the correct models/parts
        Add its value to epoch losses dictionary
        It's important that config is in conventional format
        so that each loss flow to its intended parts.

        Parameters
        batch_losses: Dict
            Dictionary of computed losses 
        """
        for loss_name, computed_loss in batch_losses.items():
            if not self.losses[loss_name]['apply']:
                continue

            if not self.current_epoch >= self.losses[loss_name]['activate_at']:
                continue

            if self.current_epoch >= self.losses[loss_name]['stop_at']:
                continue

            if computed_loss is None:
                continue

            computed_loss = self.get_weighted_loss(
                computed_loss=computed_loss,
                loss_name=loss_name
                )

            self.add_loss(
                loss_name=loss_name, 
                loss_dict=epoch_outs, 
                computed_loss=computed_loss
                )

            if isinstance(self.losses[loss_name]['gradient_flow'], list):
                self.freeze_all(_except=self.losses[loss_name]['gradient_flow'])
                cloned_gradients = self.clone_gradients()
                computed_loss.backward(retain_graph=True)
                # for m in self.models.keys():
                #    if m in self.losses[loss_name]['gradient_flow']:
                #        torch.nn.utils.clip_grad_norm_(self.models[m].parameters(), 1.0)

                self.get_gradient_by_loss(cloned_gradients, loss_name)
                self.unfreeze_all(_except=self.losses[loss_name]['gradient_flow'])
            
            elif self.losses[loss_name]['gradient_flow'].lower() == 'all':
                cloned_gradients = self.clone_gradients()
                computed_loss.backward(retain_graph=True)
                # for m in self.models.keys():
                #     torch.nn.utils.clip_grad_norm_(self.models[m].parameters(), 1.0)

                self.get_gradient_by_loss(cloned_gradients, loss_name)
            else:
                raise ValueError(f'{loss_name} gradient_flow should be either a list of strings or all')
        
        #Gradient clipping after all losses backwards
        for m in self.models.keys():
            torch.nn.utils.clip_grad_norm_(self.models[m].parameters(), 1.0)

    def clone_gradients(self):
        cloned_gradients = None
        if self.hparams['save_gradients']['apply']:
            if self.current_epoch % self.hparams['save_gradients']['interval'] == 0:
                cloned_gradients = {}
                for model_name, model in self.models.items():
                    cloned_gradients[model_name] = {
                        name: param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                        for name, param in model.named_parameters()
                    }

        return cloned_gradients

    def get_gradient_by_loss(self, cloned_gradients, loss_name):
        if self.hparams['save_gradients']['apply']:
            if self.current_epoch % self.hparams['save_gradients']['interval'] == 0:
                for model_name, model in self.models.items():
                    isolated_gradients = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            isolated_grad = param.grad - cloned_gradients[model_name][name]
                        else:
                            # Handle the case where param.grad is None
                            isolated_grad = torch.zeros_like(cloned_gradients[model_name][name])
                        isolated_gradients[name] = isolated_grad

                    isolated_gradients_norm = self.aggregate_gradient_norm(
                        isolated_gradients
                    )

                    self.gradient_by_loss[f'Gradients_by_loss {loss_name}/: {model_name}'] = isolated_gradients_norm

    def aggregate_gradient_norm(self, isolated_gradients):
        """ Compute the aggregate gradient norm from isolated gradients """
        total_norm = 0
        for grad in isolated_gradients.values():
            total_norm += grad.norm(2).item() ** 2
        return total_norm ** 0.5


    def add_prefix(self, prefix, _dict):
        """
        Add a prefix a dictionary of metrics (for wandb sections)

        Parameters
        --------------
        prefix: Str
            Prefix to add 
        _dict: Dict
            Dictionary of metrics
        """
        return {f'{prefix}{k}':v for k,v in _dict.items()}

    def load_weights(self, weights_folder_path):

        print(f'Loading weights from {weights_folder_path}')
        for model_name in self.models.keys():
            weights = torch.load(f'{weights_folder_path}/{model_name}.pt')
            self.models[model_name].load_state_dict(weights)
            self.models[model_name].eval()
