from model.trainers.base_trainer import *
import torchvision


class DehazeTrainer_pair(BaseTrainer):
    def __init__(self,cfg):
        self.cfg = cfg
        super(DehazeTrainer_pair,self).__init__(self.cfg)

        rank, world_size = link.get_rank(), link.get_world_size()
        self.rank = rank
        self.world_size = world_size
        last_iter = -1

        #Model
        model = model_entry(self.cfg.model)
        model.cuda()

        if cfg.evaluate or cfg.finetune:
            self._load_state(cfg.load_path, model)

        self.model = DistModule(model, True)
        

        #optimizer
        opt_config = self.cfg.optimizer
        opt_config.kwargs.lr = self.cfg.lr_scheduler.base_lr
        opt_config.kwargs.params = self.model.parameters()
        self.optimizer = optim_entry(opt_config)

        #lr_scheduler
        self.cfg.lr_scheduler['optimizer'] = self.optimizer
        self.cfg.lr_scheduler['last_iter'] = last_iter
        self.lr_scheduler = get_scheduler(self.cfg.lr_scheduler)
        if not self.cfg.evaluate:
            #criterion
            criterion_config = self.cfg.criterion
            self.criterion = loss_entry(criterion_config).cuda()


    def _train_model(self,train_dataset,last_iter=-1):
        train_sampler = DistributedGivenIterationSampler(train_dataset, self.cfg.lr_scheduler.max_iter, self.cfg.batch_size, 
                                                        last_iter=last_iter)
        train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.workers, pin_memory=True, sampler=train_sampler)
        
        self._train_distributed(train_loader,last_iter+1)

    def _eval_model(self,dataset,last_iter=-1):
        val_loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.workers, pin_memory=True
        )
        self._test(val_loader)

    def _train_distributed(self,loader,start_iter):

        batch_time = AverageMeter(self.cfg.print_freq)
        data_time = AverageMeter(self.cfg.print_freq)
        losses = AverageMeter(self.cfg.print_freq)

        self.model.train()

        world_size = link.get_world_size()
        rank = link.get_rank()

        end = time.time()

        curr_step = start_iter
        tmp_step = start_iter
        for iter_id, (src, target,_,_) in enumerate(loader):
            curr_step = curr_step + 1
            tmp_step = tmp_step + 1
            current_lr = self.lr_scheduler.get_lr()[0]
            data_time.update(time.time() - end)

            target = target.cuda()
            src = src.cuda()

            output = self.model(src,target.cuda())
            output = torch.clamp(output,0,1)
            loss = self.criterion(src,target,output) / world_size

            reduced_loss = loss.clone()

            link.allreduce(reduced_loss)

            losses.update(reduced_loss.item())

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            #self._debug()
            reduce_gradients(self.model, True)
            self.optimizer.step()
            self.lr_scheduler.step(curr_step)

            # measure elapsed time
            batch_time.update(time.time() - end)

            if tmp_step % self.cfg.print_freq == 0 and rank == 0:
                self.tb_logger.add_scalar('loss_train', losses.avg, tmp_step)
                self.tb_logger.add_scalar('lr', current_lr, tmp_step)
                self.logger.info('Iter: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'LR {lr:.4f}'.format(
                    tmp_step, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, lr=current_lr))

            if tmp_step > 0 and tmp_step % self.cfg.save_freq == 0 and rank == 0:
                with torch.no_grad():
                    rec = self.model(src,target,rec=True)
                output_name = os.path.join(self.cfg.save_path,'imgs',str(iter_id)+'.jpg')
                output_images = torch.cat((src.detach().cpu(), target.detach().cpu(), output.detach().cpu(),rec.detach().cpu()), 0)
                torchvision.utils.save_image(output_images, output_name, nrow=1)
                self._save_checkpoint(tmp_step,self.cfg.model.arch,self.cfg.save_path+'/ckpt/'+str(tmp_step))

            link.barrier()
            end = time.time()

    
    @torch.no_grad()
    def _test(self,loader):
        from tqdm import tqdm
        for iter_id, (src, target, _, _, src_path, target_path) in tqdm(enumerate(loader)):

            target = target.cuda()
            src = src.cuda()

            output = self.model(src,target)
            output = torch.clamp(output,0,1)
            for idx in range(len(output)):
                output_name = os.path.join(self.cfg.save_path,self.cfg.pred_path,'cat',src_path[idx]+'_'+target_path[idx])
                output_images = torch.cat((src[idx].detach().cpu().unsqueeze(0), target[idx].detach().cpu().unsqueeze(0), output[idx].detach().cpu().unsqueeze(0)), 0)
                torchvision.utils.save_image(output_images, output_name, nrow=1)

                pred_name = os.path.join(self.cfg.save_path,self.cfg.pred_path,'pred',src_path[idx])
                torchvision.utils.save_image(output[idx].detach().cpu().unsqueeze(0), pred_name)

    