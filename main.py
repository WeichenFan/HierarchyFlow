import os
import yaml
import torch
import argparse
import logging
import shutil
import numpy as np
import random
from easydict import EasyDict
import spring.linklink as link

from model.trainers import *
from model.data import get_dataset_entry
from model.utils.distributed_utils import dist_init

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    rank, world_size = dist_init()
    with open(args.config) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    config = EasyDict(config['common'])
    name = args.config.split('/')[-1].split('.')[0]
    config.save_path = os.path.join(args.save_path,name)
    config.load_path = args.load_path
    config.evaluate = args.evaluate
    config.finetune = args.finetune


    if rank == 0:
        print('evaluate: ',config.evaluate)
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
            print('mkdir save_path')

        shutil.copyfile(args.config,os.path.join(config.save_path,'config.yaml'))
        if not os.path.exists(os.path.join(config.save_path,'events')):
            os.makedirs(os.path.join(config.save_path,'events'))
            print('mkdir events_path')

        if not os.path.exists(os.path.join(config.save_path,'logs')):
            os.makedirs(os.path.join(config.save_path,'logs'))
            print('mkdir logs_path')

        if not os.path.exists(os.path.join(config.save_path,'ckpt')):
            os.makedirs(os.path.join(config.save_path,'ckpt'))
            print('mkdir ckpt_path')

        if not os.path.exists(os.path.join(config.save_path,'imgs')):
            os.makedirs(os.path.join(config.save_path,'imgs'))
            print('mkdir imgs_path')
            
    if args.seed != -1:
        config['seed'] = args.seed

    set_random_seed(config['seed'])

    dataset = get_dataset_entry(config)

    if not config.evaluate:
        trainer = globals()[config.trainer](config)
        trainer._train_model(dataset,last_iter=-1)
    else:
        if not os.path.exists(os.path.join(config.save_path,args.pred_path,'pred')):
            os.makedirs(os.path.join(config.save_path,args.pred_path,'pred'))
            print('mkdir imgs_path')
        if not os.path.exists(os.path.join(config.save_path,args.pred_path,'cat')):
            os.makedirs(os.path.join(config.save_path,args.pred_path,'cat'))
            print('mkdir imgs_path')
        config.pred_path = args.pred_path

        trainer = globals()[config.trainer](config)
        trainer._eval_model(dataset,last_iter=-1)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchy Flow')
    parser.add_argument('--config', default='../config/cash/cash_osr.yaml')
    parser.add_argument('--save_path', default='./')
    parser.add_argument('--pred_path', default='./')
    parser.add_argument('--load_path', default='', type=str)
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--seed', default=-1)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--fake', action='store_true')
    parser.add_argument('--fuse-prob', action='store_true')
    parser.add_argument('--fusion-list', nargs='+', help='multi model fusion list')
    args = parser.parse_args()
    main(args)
    link.finalize()