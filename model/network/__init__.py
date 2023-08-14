
from .HFnet import hf_net
from .HFnet_origin import hf_net_ori
from .HFnet_De import hf_net_de
from .Glow_De import glow_net_de

def model_entry(config):
    return globals()[config['arch']](**config['kwargs'])