###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model
from .data_augmentation import Cutout, Cutmix, PatchGaussian #data_aug

from .option import Options

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

class Trainer():
    def __init__(self, args):
        self.args = args
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])

        # data_aug
        if args.data_aug == 'Cutout':
            input_transform.transforms.append(Cutout())
        elif args.data_aug == 'PatchGaussian':
            input_transform.transforms.append(PatchGaussian())
        self.use_Cutmix = (args.data_aug == 'Cutmix')

        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size, 'root': '/scratch/yz8458/.encoding/data'}
        trainset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train',
                                           **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode ='val',
                                           **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        # model
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = SyncBatchNorm,
                                       base_size = args.base_size, crop_size = args.crop_size, root = '/scratch/yz8458/.encoding/models')
        # print(model)
        print(sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6)
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'jpu'):
            params_list.append({'params': model.jpu.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
        optimizer = torch.optim.SGD(params_list, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
        # criterions
        self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux,
                                            nclass=self.nclass,
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight)
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()
        # resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            # data_aug
            if self.use_Cutmix:
                image, target = Cutmix(image, target)

            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            outputs = self.model(image)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best, filename='checkpoint_{}.pth.tar'.format(epoch))

        train_loss = train_loss / len(self.trainloader)
        print('==>Epoches %i, Train loss: %.3f\n' % (epoch, train_loss))
        return train_loss


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        mIoU_avg, pixAcc_avg = 0.0, 0.0
        for i, (image, target) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            mIoU_avg, pixAcc_avg = mIoU_avg + mIoU, pixAcc_avg + pixAcc
            # tbar.set_description(
            #     'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        new_pred = (pixAcc + mIoU)/2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': new_pred,
        }, self.args, is_best)
        pixAcc_avg = pixAcc_avg / len(self.valloader)
        mIoU_avg = mIoU_avg / len(self.valloader)
        print('==>Epoches %i, pixAcc_avg: %.3f, mIoU_avg: %.3f\n' % (epoch, pixAcc_avg, mIoU_avg))
        return pixAcc_avg, mIoU_avg


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        train_loss = trainer.training(epoch)
        if not trainer.args.no_val:
            pixAcc, mIoU = trainer.validation(epoch)
            history_str = '%.3f,%.3f,%.3f\n' % (train_loss, pixAcc, mIoU)
            utils.save_history(history_str, args.checkname)

