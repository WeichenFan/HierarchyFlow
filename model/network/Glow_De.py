import torch
import torch.nn as nn
from model.layers import Glow


class GLOW_de(nn.Module):
    def __init__(self,channel_list,n_block,**kwargs):
        super(GLOW_de, self).__init__()

        self.glow = Glow(in_channel=3,n_flow=16,n_block=2,affine=False)

    def forward(self, input, z_list):
        
        log_p_sum, logdet, z_outs = self.glow.forward(input)
        rec = self.glow.reverse(z_list)

        return log_p_sum, logdet, z_outs, rec


def glow_net_de(*args, **kwargs):
    print('using Glow for density estimation')
    model = GLOW_de(*args, **kwargs)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('num: ',num_params/ 1e6,' M')

    return model
