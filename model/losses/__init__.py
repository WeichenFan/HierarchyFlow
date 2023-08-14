from .VGG_loss import vgg_loss
from .Dehaze_loss import dehaze_loss

def loss_entry(config):
    return globals()[config['type']](**config['kwargs'])
