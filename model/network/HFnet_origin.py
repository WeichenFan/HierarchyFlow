import torch
import torch.nn as nn
from model.layers import HierarchyFilter_ORI,StyleEncoder


class HierarchyFlow(nn.Module):
    def __init__(self,channel_list,n_block,**kwargs):
        super(HierarchyFlow, self).__init__()

        self.padding = torch.nn.ReflectionPad2d(10)
        self.blocks = nn.ModuleList()
        for i in range(n_block - 1):
            self.blocks.append(HierarchyFilter_ORI(in_channel=channel_list[i], out_channel=channel_list[i+1],**kwargs))
            
        self.net = StyleEncoder(n_downsample=2,input_dim=3, dim=64, style_dim=8, norm='none', activ='relu', pad_type='reflect')

    def forward(self,input,style,rec=False):
        
        style_code = self.net(style)

        input = self.padding(input)
        _, _, height, width = input.shape
        
        for block in self.blocks:
            input = block(input)

        for _, block in enumerate(self.blocks[::-1]):
            input = block.reverse(input, style_code, rec)

        input = input[:,:,10:height-10,10:width-10]
        return input


def hf_net_ori(*args, **kwargs):
    print('using HierarchyFlow')
    model = HierarchyFlow(*args, **kwargs)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('num: ',num_params/ 1e6,' M')

    return model
