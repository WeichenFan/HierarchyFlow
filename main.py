import argparse
import os
import yaml
import shutil
import random
from easydict import EasyDict
import torch
import torch.distributed as dist

from model.trainers.hf_trainer import Trainer


parser = argparse.ArgumentParser(description='PyTorch HierarchyFlow Training')
parser.add_argument('--config', type=str, default='configs/config.yaml', help='config file')
parser.add_argument('--eval_only', action='store_true', help='evaluation mode')
parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
parser.add_argument('--seed', type=int, default=0, help='seed for initializing training')
parser.add_argument('--load_path', type=str, help='path for ckpt')

def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    print(args.nprocs, args.local_rank)
    set_random_seed(args.seed)

    dist.init_process_group(backend='nccl')
    torch.backends.cudnn.benchmark = True

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(cfg)
    cfg.eval_mode = args.eval_only
    if 'k' in args:
        cfg.loss.k = args.k
    if 'load_path' in args:
        cfg.load_path = args.load_path
        
    if args.local_rank == 0:
        print(cfg)
        if not os.path.exists(cfg.output):
            os.makedirs(cfg.output)
        if not os.path.exists(os.path.join(cfg.output, cfg.task_name)):
            os.makedirs(os.path.join(cfg.output, cfg.task_name))
        if not os.path.exists(os.path.join(cfg.output, cfg.task_name, 'img_save')):
            os.makedirs(os.path.join(cfg.output, cfg.task_name, 'img_save'))
        if not os.path.exists(os.path.join(cfg.output, cfg.task_name, 'model_save')):
            os.makedirs(os.path.join(cfg.output, cfg.task_name,'model_save'))
        if not os.path.exists(os.path.join(cfg.output, cfg.task_name, 'eval_results')):
            os.makedirs(os.path.join(cfg.output, cfg.task_name,'eval_results'))
            os.makedirs(os.path.join(cfg.output, cfg.task_name,'eval_results', 'pred'))
            os.makedirs(os.path.join(cfg.output, cfg.task_name,'eval_results', 'cat_img'))
        shutil.copy(os.path.join(args.config), os.path.join(cfg.output, cfg.task_name, 'cfg.yaml'))

    trainer = Trainer(cfg, args.local_rank, args.nprocs)

    if args.eval_only:
        trainer.eval()
        return

    trainer.train()


if __name__ == "__main__":
    main()