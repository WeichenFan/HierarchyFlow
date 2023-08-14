
import os
import time
import torch
import logging
import random
import datetime
import pprint
import numpy as np
import spring.linklink as link
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from model.utils import (dist_init,simple_group_split,DistModule,
                        DistributedGivenIterationSampler,DistributedTestSampler,
                        AverageMeter,reduce_gradients)
from model.network import model_entry
from model.losses import loss_entry
from model.optim import optim_entry,get_scheduler

def logger_config(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

class BaseTrainer():
    def __init__(self, cfg):
        self.__set_random_seed(cfg['seed'])
        log_folder = os.path.join(cfg['save_path'],'logs')
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M%S')
        self.logger = logger_config(log_folder, cfg.save_path+'/logs/'+now+'.txt')
        if not cfg.evaluate:
            self.tb_logger = SummaryWriter(cfg.save_path+'/events')

        if link.get_rank() == 0:
            self.logger.info('config: {}'.format(pprint.pformat(cfg)))

    def __set_random_seed(self,seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _adv_image(self, net, inputs, target, criterion, world_size, targets=None, eps=0.07):
        net.eval()
        images = inputs.clone()
        images.requires_grad = True

        output = net(images)
        loss = criterion(output, target, input=images) / world_size
        loss.backward()

        adv_inputs = images + eps*images.grad.sign()
        net.train()
        return adv_inputs.detach()

    def _train_model(self,train_dataset,val_dataset,last_iter=-1):
        pass
    def _eval_model(self,train_dataset,val_dataset,last_iter=-1):
        pass
    def _train(self,loader):
        pass

    def _validate(self,loader):
        pass

    def _debug(self):
        for name,param in self.model.named_parameters():
            try:
                d = param.grad.data
            except:
                print('name: ',name)

    def _save_checkpoint(self,tmp_step,arch,path):
        self.model.eval()
        def save_checkpoint(state, filename):
            torch.save(state, filename+'.pth.tar')
        save_checkpoint({
            'step':tmp_step,
            'arch':arch,
            'state_dict':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict()
            },path)
        self.model.train()

    def _load_state(self, path, model, optimizer=None):

        rank = link.get_rank()

        def map_func(storage, location):
            return storage.cuda()

        if os.path.isfile(path):
            if rank == 0:
                print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path, map_location=map_func)
            new_dict = {}
            sd = checkpoint['state_dict']
            for k in sd.keys():
                new_k = k.split('module.')[1]
                new_dict[new_k] = sd[k]

            model.load_state_dict(new_dict, strict=False)

            if rank == 0:
                ckpt_keys = set(new_dict.keys())
                own_keys = set(model.state_dict().keys())
                missing_keys = own_keys - ckpt_keys
                for k in missing_keys:
                    print('caution: missing keys from checkpoint {}: {}'.format(path, k))

            if optimizer != None:
                best_prec1 = checkpoint['best_prec1']
                last_iter = checkpoint['step']
                #optimizer.load_state_dict(checkpoint['optimizer'])
                if rank == 0:
                    print("=> also loaded optimizer from checkpoint '{}' (iter {})".format(path, last_iter))
                return best_prec1, last_iter
        else:
            if rank == 0:
                print("=> no checkpoint found at '{}'".format(path))