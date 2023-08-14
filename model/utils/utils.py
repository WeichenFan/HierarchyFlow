import os
import logging
import shutil
import torch
from torch.utils.data.sampler import Sampler
import spring.linklink as link
import math
import numpy as np
from collections import defaultdict

def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank//group_size]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count

class DistributedTestSampler(Sampler):
    def __init__(self, dataset, world_size, rank, validation=False):
        num_total = len(dataset)
        part = math.ceil(num_total / world_size)
        if rank == world_size - 1:
            self.num_samples = num_total - (part * (world_size - 1))
            self.indices = range(part * (world_size - 1), num_total)
            if validation:
                self.indices = list(self.indices) + list(range(part-self.num_samples))
        else:
            self.num_samples = part
            self.indices = range(part * rank, part * (rank + 1))

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        return iter(self.indices)
        

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        world_size (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within world_size.
    """

    def __init__(self, dataset, world_size=None, rank=None, round_up=True):
        if world_size is None:
            world_size = link.get_world_size()
        if rank is None:
            rank = link.get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.round_up = round_up
        self.epoch = 0
        
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        if self.round_up:
            self.total_size = self.num_samples * self.world_size
        else:
            self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        if self.round_up or (not self.round_up and self.rank < self.world_size-1):
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1):
        if world_size is None:
            world_size = link.get_world_size()
        if rank is None:
            rank = link.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter*self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1

        ind_list = list(indices)
        for i in range(num_repeat):
            np.random.shuffle(indices)
            ind_list += list(indices)
        indices = np.array(ind_list)

        #indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg+self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        #return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size

def load_state(path, model, optimizer=None):

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

        model.load_state_dict(new_dict, strict=True)

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

def param_group_no_wd(model):
    pgroup_no_wd = []
    names_no_wd = []
    pgroup_normal = []

    type2num = defaultdict(lambda : 0)
    for name,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1
        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1
        elif (isinstance(m, torch.nn.BatchNorm2d) 
              or isinstance(m, torch.nn.BatchNorm1d) 
              or isinstance(m, link.nn.SyncBatchNorm2d)):
            if m.weight is not None:
                pgroup_no_wd.append(m.weight)
                names_no_wd.append(name+'.weight')
                type2num[m.__class__.__name__+'.weight'] += 1
            if m.bias is not None:
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1

    for name,p in model.named_parameters():
        if not name in names_no_wd:
            pgroup_normal.append(p)

    return [{'params': pgroup_normal}, {'params': pgroup_no_wd, 'weight_decay': 0.0}], type2num
