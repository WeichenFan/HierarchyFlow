import torch
import torch.nn as nn
from model.layers import HierarchyFilter_DE


class HierarchyFlow_de(nn.Module):
    def __init__(self,channel_list,n_block,**kwargs):
        super(HierarchyFlow_de, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(n_block - 1):
            self.blocks.append(HierarchyFilter_DE(in_channel=channel_list[i], out_channel=channel_list[i+1],**kwargs))
            


    def forward(self, input, z_list):
        
        _, _, height, width = input.shape
        z_outs = []
        log_p_sum = 0

        for block in self.blocks:
            input, log_p, feature = block(input)
            z_outs.append(feature)
            log_p_sum = log_p_sum + log_p

        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                output = block.reverse(z_list[-1], z_list[-1])

            else:
                output = block.reverse(output, z_list[-(i + 1)])

        return z_outs,log_p_sum,output


def hf_net_de(*args, **kwargs):
    print('using HierarchyFlow for density estimation')
    model = HierarchyFlow_de(*args, **kwargs)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('num: ',num_params/ 1e6,' M')

    return model
