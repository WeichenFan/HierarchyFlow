from torch.optim import SGD
from torch.optim import Adam
from torch.optim import AdamW
from .scheduler import *

def optim_entry(config):
    return globals()[config['type']](**config['kwargs'])

def get_scheduler(config):
    config = EasyDict(config)

    if config.type == 'STEP':
        return StepLRScheduler(config.optimizer, config.lr_steps, config.lr_mults, config.base_lr, config.warmup_lr, config.warmup_steps, last_iter=config.last_iter)
    elif config.type == 'COSINE':
        return cosineLR(config.optimizer,config.T_max,config.eta_min, config.base_lr, config.warmup_lr, config.warmup_steps, last_iter=config.last_iter)
        #return CosineLRScheduler(config.optimizer, config.max_iter, config.min_lr, config.base_lr, config.warmup_lr, config.warmup_steps, last_iter=config.last_iter)
    else:
        raise RuntimeError('unknown lr_scheduler type: {}'.format(config.type))