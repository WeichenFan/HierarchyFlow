import numpy as np
import os
from numpy.lib.histograms import histogram


import torch
import torch.distributed as dist
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from model.losses.vgg_loss import VGGLoss
from model.network.hf import HierarchyFlow
from model.utils.dataset import get_dataset
from model.utils.sampler import DistributedGivenIterationSampler, DistributedTestSampler
from tensorboardX import SummaryWriter

import logging
from model.utils.log_helper import init_log

init_log('pytorch hierarchy flow')
global_logger = logging.getLogger('pytorch hierarchy flow')


def save_checkpoint(state, filename):
    torch.save(state, filename+'.pth.tar')

def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['step']

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class Trainer():
    def __init__(self, cfg, local_rank, world_size):
        self.cfg = cfg
        self.rank = local_rank
        self.world_size = world_size
        
        model = HierarchyFlow(self.cfg.network.pad_size, self.cfg.network.in_channel, self.cfg.network.out_channels, self.cfg.network.weight_type)
        model.cuda(self.rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])

        if self.rank == 0:
            global_logger.info(self.cfg)
            global_logger.info(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)

        if self.cfg.eval_mode or (self.cfg.resume and os.path.isfile(self.cfg.load_path)):
            self.model, self.optimizer, self.resumed_step = load_checkpoint(self.cfg.load_path, model, optimizer)
            global_logger.info("=> loaded checkpoint '{}' with current step {}".format(self.cfg.load_path, self.resumed_step))
        else:
            self.model = model
            self.optimizer = optimizer
            self.resumed_step = -1

        if self.cfg.lr_scheduler.type == 'cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.max_iter, self.cfg.lr_scheduler.eta_min)
        else:
            raise RuntimeError('lr_scheduler {} is not implemented'.format(self.cfg.lr_scheduler))

        self.criterion = VGGLoss(self.cfg.loss.vgg_encoder).cuda(self.rank)
        
        if self.rank == 0:
            self.logger = SummaryWriter(os.path.join(self.cfg.output, self.cfg.task_name, 'runs'))

    def train(self):
        train_dataset = get_dataset(self.cfg.dataset.train)
        train_sampler = DistributedGivenIterationSampler(train_dataset,
            self.cfg.max_iter, self.cfg.dataset.train.batch_size, world_size=self.world_size, rank=self.rank, last_iter=-1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.dataset.train.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            sampler=train_sampler)

        for batch_id, batch in enumerate(train_loader):
            self.train_iter(batch_id, batch)
        
        self.eval()

    def eval(self):
        test_dataset = get_dataset(self.cfg.dataset.test)
        test_sampler = DistributedTestSampler(test_dataset, world_size=self.world_size, rank=self.rank)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.cfg.dataset.test.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=False, 
            sampler=test_sampler)
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                content_images = batch[0].cuda(self.rank)
                style_images = batch[1].cuda(self.rank)
                names = batch[2]
                outputs = self.model(content_images, style_images)
                outputs = torch.clamp(outputs, 0, 1)
                outputs = outputs.cpu()

                for idx in range(len(outputs)):
                    output_name = os.path.join(self.cfg.output, self.cfg.task_name, 'eval_results', 'pred', names[idx])
                    save_image(outputs[idx].unsqueeze(0), output_name)
                    if idx == 0:
                        output_name = os.path.join(self.cfg.output, self.cfg.task_name, 'eval_results', 'cat_img', names[idx])
                        output_images = torch.stack((content_images[idx].cpu(), style_images[idx].cpu(), outputs[idx]), 0)
                        save_image(output_images, output_name, nrow=1)
                if self.rank == 0 and batch_id % 10 == 1:
                    global_logger.info('predicting {}th batch...'.format(batch_id))
        if self.rank == 0:
            global_logger.info('Save predictions to {}\nDone.'.format(os.path.join(self.cfg.output, self.cfg.task_name, 'eval_results')))

    def train_iter(self, batch_id, batch):
        content_images = batch[0].cuda(self.rank)
        style_images = batch[1].cuda(self.rank)

        outputs = self.model(content_images, style_images)
        outputs = torch.clamp(outputs, 0, 1)

        loss_c, loss_s = self.criterion(content_images, style_images, outputs, self.cfg.loss.k)
        loss_c = loss_c.mean()
        loss_s = loss_s.mean()
        loss = loss_c + self.cfg.loss.weight * loss_s

        torch.distributed.barrier()

        loss = reduce_mean(loss, self.world_size)
        loss_c = reduce_mean(loss_c, self.world_size)
        loss_s = reduce_mean(loss_s, self.world_size)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        if self.rank == 0:
            current_lr = self.lr_scheduler.get_lr()[0]
            self.logger.add_scalar("current_lr", current_lr, batch_id + 1)
            self.logger.add_scalar("loss_c", loss_c.item(), batch_id + 1)
            self.logger.add_scalar("loss_s", loss_s.item(), batch_id + 1)
            self.logger.add_scalar("loss", loss.item(), batch_id + 1)

        if self.rank == 0 and batch_id % self.cfg.print_freq == 0:
            global_logger.info('batch: {}, style_loss: {}, content_loss: {}, loss: {}'.format(batch_id, loss_s.item(), loss_c.item(), loss.item()))
            output_name = os.path.join(self.cfg.output, self.cfg.task_name, 'img_save', str(batch_id)+'.jpg')
            output_images = torch.cat((content_images.cpu(), style_images.cpu(), outputs.cpu()), 0)
            save_image(output_images, output_name, nrow=1)

        if self.rank == 0 and batch_id % self.cfg.save_freq == 0:
            save_checkpoint({
                'step':batch_id,
                'state_dict':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict()
                },os.path.join(self.cfg.output, self.cfg.task_name, 'model_save', str(batch_id)+ '.ckpt'))