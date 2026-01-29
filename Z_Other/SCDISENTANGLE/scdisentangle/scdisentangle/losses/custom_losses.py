import torch
import torch.nn as nn

class CustomLosses:
    """
    Losses used as criterions should be defined here 
    and pointed out in the config file
    """
    def __init__(self, device, **kwargs):
        super(CustomLosses, self).__init__()
        
        self.kwargs = kwargs
        self.device = device
        
    def nb_loss(self, forward_outs, loss_dict):
        px = forward_outs[loss_dict['pred_key']]
        target = forward_outs[loss_dict['gt_key']]

        return -px.log_prob(target).sum(dim=1).sum() #.mean() #.sum()

    def recover_latent(self, forward_outs, loss_dict):

        latent = forward_outs[loss_dict['latent_key']]
        post_latent = forward_outs[loss_dict['post_latent_key']]
       

        loss = ((latent - post_latent)**2).sum() #.sum(dim=1).mean() #mean()#.sum()

        return loss