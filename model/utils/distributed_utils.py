import os
import time
import torch
import spring.linklink as link

class DistModule(torch.nn.Module):
    def __init__(self, module, sync=False):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)

        if not sync:
            self._grad_accs = []
            self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)

    def _register_hooks(self):
        for i,(name,p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(self, name, p, i):
        def hook(*ignore):
            link.allreduce_async(name, p.grad.data)
        return hook

def reduce_gradients(model, sync=False):
    """ average gradients """
    if sync:
        for name, param in model.named_parameters():
            if param.requires_grad:
                link.allreduce(param.grad.data)
    else:
        link.synchronize()

def broadcast_params(model):
    """ broadcast model parameters """
    for name,p in model.state_dict().items():
        link.broadcast(p, 0)

def dist_init():
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id%num_gpus)

    link.initialize()
    world_size = link.get_world_size()
    rank = link.get_rank()
    # link.initialize(num_devices=1)
    # torch.cuda.set_device(link.get_device_id(0))
    # torch.backends.cudnn.benchmark = True
    # world_size = link.get_world_size()
    # rank = link.get_rank()
    return rank, world_size
