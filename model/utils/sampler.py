import math
import numpy as np
from torch.utils.data.sampler import Sampler


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

class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1):
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
        indices = np.tile(indices, num_repeat)
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