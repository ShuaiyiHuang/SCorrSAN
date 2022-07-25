r'''
    modified training script of GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''

import argparse
import os
import pickle
import random
import time
from os import path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader

from data import download
import utils_training.optimize as optimize
from utils_training.evaluation import Evaluator
from utils_training.utils import parse_list, save_checkpoint, boolean_string, load_checkpoint
from utils_training.logger import Logger
torch.multiprocessing.set_sharing_strategy('file_system')

from models.model.scorrsan import SCorrSAN

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='SCorrSAN Training Script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
                        
    parser.add_argument('--datapath', type=str, default='../Datasets_CATs')
    parser.add_argument('--benchmark', type=str, default='spair', choices=['pfpascal', 'spair'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                        help='learning rate (default: 3e-5)')
    parser.add_argument('--lr-backbone', type=float, default=3e-6, metavar='LR',
                        help='learning rate (default: 3e-6)')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--step', type=str, default='[150]')
    parser.add_argument('--step_gamma', type=float, default=0.5)
    parser.add_argument('--augmentation', type=boolean_string, nargs='?', const=True, default=True)

    # SCorrSAN
    parser.add_argument('--pseudo_ksize', type=int, default = 7, help='ksize for dilation when enlarge valid mask region for pseudo-loss, used in co-sup') # 20220221
    parser.add_argument('--remember_rate', type=float, help='remember rate', default=0.2)
    parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')

    parser.add_argument('--lmd_loss_pseudo', type=float, default=10.0, help='pseudo loss weight')
    parser.add_argument('--pstart_ep', type=int, default=4, help='>= which epoch use lmd_loss_pseudo, smaller than this epoch pseudo loss is zero')

    parser.add_argument('--sce_ksize', type = int, default = 7, help = 'spatial context encoder spatial size')
    parser.add_argument('--sce_outdim', type = int, default = 2048, help = 'spatial context encoder output channels')
    
  
    # Seed
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize logger 
    Logger.initialize(args)

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)
    train_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'trn', args.augmentation,)
    val_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'val', False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=False)

    def initialize_model_status(args):

        model_ = SCorrSAN(sce_ksize = args.sce_ksize, sce_outdim = args.sce_outdim)
        
        param_model_ = [param for name, param in model_.named_parameters() if 'feature_extraction' not in name]
        param_backbone_ = [param for name, param in model_.named_parameters() if 'feature_extraction' in name]

        # Optimizer
        optimizer_ = optim.AdamW([{'params': param_model_, 'lr': args.lr}, {'params': param_backbone_, 'lr': args.lr_backbone}], weight_decay=args.weight_decay)
        
        # Scheduler
        scheduler_ = lr_scheduler.CosineAnnealingLR(optimizer_, T_max=args.epochs, eta_min=1e-6, verbose=True) if args.scheduler == 'cosine' else lr_scheduler.MultiStepLR(optimizer_, milestones=parse_list(args.step), gamma=args.step_gamma, verbose=True)
            
        return model_, optimizer_, scheduler_
        
    model, optimizer, scheduler = initialize_model_status(args)
    model_tc, optimizer_tc, scheduler_tc = initialize_model_status(args)
   
    remember_rate_schedule = np.ones(args.epochs)*0.9
    remember_rate_schedule[:args.num_gradual] = np.linspace(args.remember_rate, 0.9, args.num_gradual)

    if not os.path.isdir(args.snapshots):
        os.mkdir(args.snapshots)

    cur_snapshot = args.name_exp
    if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
        os.makedirs(osp.join(args.snapshots, cur_snapshot))

    with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    best_val_pck = 0
    start_epoch = 0
    best_epoch = 0

    best_val_pck_tc = 0.0
    best_epoch_tc = 0

    best_val_pck_merge = 0.0
    best_epoch_merge = 0
    best_model_type = 'Undefined'

    # load from latest model
    if os.path.exists(osp.join(args.snapshots, args.name_exp, 'model_latest.pth')):
        model, optimizer, scheduler, start_epoch, best_val_pck, best_epoch, best_val_pck_merge, best_epoch_merge, best_model_type = load_checkpoint(model = model, optimizer= optimizer, scheduler = scheduler, filename = osp.join(args.snapshots, args.name_exp, 'model_latest.pth'))
        model_tc, optimizer_tc, scheduler_tc, start_epoch, best_val_pck_tc, best_epoch_tc, best_val_pck_merge, best_epoch_merge, best_model_type = load_checkpoint(model = model_tc, optimizer= optimizer_tc, scheduler = scheduler_tc, filename = osp.join(args.snapshots, args.name_exp, 'model_tc_latest.pth'))
        Logger.info('#######Load from the latest model done, start epo {}, best_val_pck_merge {}, best_epo_merge {}, best_test_pck_merge {}#########\n'.format(start_epoch, best_val_pck_merge, best_epoch_merge, best_test_pck_merge))

    # create summary writer
    save_path=osp.join(args.snapshots, cur_snapshot)
  
    train_writer = SummaryWriter(os.path.join(save_path, 'tb'))
    test_writer = train_writer

    model = nn.DataParallel(model)
    model = model.to(device)

    model_tc = nn.DataParallel(model_tc)
    model_tc = model_tc.to(device)

    train_started = time.time()

    for epoch in range(start_epoch, args.epochs):
        trn_ep_st = time.time()

        scheduler.step(epoch)

        scheduler_tc.step(epoch)

        train_loss, train_loss_tc = optimize.train_epoch(model, optimizer, train_dataloader, device, epoch, train_writer, lmd_loss_pseudo=args.lmd_loss_pseudo, Logger=Logger, net_tc=model_tc, optimizer_tc=optimizer_tc, remember_rate = remember_rate_schedule[epoch-args.pstart_ep],pseudo_ksize=args.pseudo_ksize, pstart_epoch = args.pstart_ep)
     
        train_writer.add_scalar('train loss', train_loss, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        train_writer.add_scalar('learning_rate_backbone', scheduler.get_lr()[1], epoch)
      
        Logger.info(colored('==> ', 'green') + 'Epoch {} Train average loss: {}'.format(epoch, train_loss))

        val_loss_grid, val_mean_pck = optimize.validate_epoch(model, val_dataloader, device, epoch=epoch,Logger = Logger, mode='Val')

        Logger.info(colored('==> ', 'blue') + 'Epoch {}, Val average grid loss : {}, Val mean PCK : {}'.format(epoch, val_loss_grid, val_mean_pck))

        test_writer.add_scalar('val PCK', val_mean_pck, epoch)
        test_writer.add_scalar('val loss', val_loss_grid, epoch)

        # tc trn logs
        train_writer.add_scalar('train loss tc', train_loss_tc, epoch)
        train_writer.add_scalar('learning_rate tc', scheduler_tc.get_lr()[0], epoch)
        train_writer.add_scalar('learning_rate_backbone tc', scheduler_tc.get_lr()[1], epoch)
        Logger.info(colored('==> ', 'green') + 'Epoch {}, Train average loss tc: {}'.format(epoch, train_loss_tc))

        # tc eval
        val_loss_grid_tc, val_mean_pck_tc = optimize.validate_epoch(model_tc, val_dataloader, device, epoch=epoch, Logger=Logger, mode='Val')
        
        Logger.info(colored('==> ', 'blue') + 'Epoch {}, Val tc average grid loss : {}, Val mean PCK tc is {}'.format(epoch, val_loss_grid_tc, val_mean_pck_tc))

        test_writer.add_scalar('val PCK tc', val_mean_pck_tc, epoch)
        test_writer.add_scalar('val loss tc', val_loss_grid_tc, epoch)

        # update tc best
        is_best_tc = val_mean_pck_tc > best_val_pck_tc
        if is_best_tc:
            best_epoch_tc = epoch
            best_val_pck_tc = val_mean_pck_tc

        # update stu best
        is_best_stu = val_mean_pck > best_val_pck
        if is_best_stu:
            best_epoch = epoch
            best_val_pck = val_mean_pck

        # merge best model
        # merge best from tea and stu
        if best_val_pck_tc>best_val_pck:
            is_best = best_val_pck_tc > best_val_pck_merge

            if is_best:
                best_val_pck_merge = best_val_pck_tc
                best_epoch_merge = best_epoch_tc
                best_model_type = 'teacher'
  
                save_checkpoint({'epoch': epoch,
                                'state_dict': model_tc.module.state_dict(),
                                'optimizer': optimizer_tc.state_dict(),
                                'scheduler': scheduler_tc.state_dict(),
                                'best_epoch': best_epoch_tc,
                                'best_val_pck': best_val_pck_tc,
                                'best_val_pck_merge': best_val_pck_merge,
                                'best_epoch_merge': best_epoch_merge,
                                'best_model_type':best_model_type,  
                                },
                                True, save_path, 'model_best_tc.pth')
        else:
            is_best = best_val_pck > best_val_pck_merge
            if is_best:
                best_val_pck_merge = best_val_pck
                best_epoch_merge = best_epoch
                best_model_type = 'student'

                save_checkpoint({'epoch': epoch,
                                'state_dict': model.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'best_epoch': best_epoch,
                                'best_val_pck': best_val_pck,
                                'best_val_pck_merge': best_val_pck_merge,
                                'best_epoch_merge': best_epoch_merge,
                                'best_model_type':best_model_type, 
                                },
                                True, save_path, 'model_best_stu.pth'.format(epoch))

        #save model latest and model_tc latest at every epoch
        save_checkpoint({'epoch': epoch,
                                'state_dict': model.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'best_epoch': best_epoch,
                                'best_val_pck': best_val_pck,
                                'best_val_pck_merge': best_val_pck_merge,
                                'best_epoch_merge': best_epoch_merge,
                                'best_model_type':best_model_type,
                                },
                                False, save_path, 'model_latest.pth')
                    
        save_checkpoint({'epoch': epoch,
                                'state_dict': model_tc.module.state_dict(),
                                'optimizer': optimizer_tc.state_dict(),
                                'scheduler': scheduler_tc.state_dict(),
                                'best_epoch': best_epoch_tc,
                                'best_val_pck': best_val_pck_tc,
                                'best_val_pck_merge': best_val_pck_merge,
                                'best_epoch_merge': best_epoch_merge,
                                'best_model_type':best_model_type, 
                                },
                                False, save_path, 'model_tc_latest.pth')

        # write best msg tb
        test_writer.add_scalar('best val pck', best_val_pck_merge, epoch)
        test_writer.add_scalar('best epoch', best_epoch_merge, epoch)
            
        # summary
        msg_stu = 'Epoch{} stu\tTrain_loss {:.6f}\tVal_loss {:.6f}\tVal_pck {:.6f}\tBest epoch {}, best_val_pck {:.3f}\n'.format \
                        (epoch, train_loss, val_loss_grid, val_mean_pck, best_epoch, best_val_pck)
        msg_tc = 'Epoch{} tc\tTrain_loss {:.6f}\tVal_loss {:.6f}\tVal_pck {:.6f}\tBest epoch {}, best_val_pck {:.3f}\n'.format \
                        (epoch, train_loss_tc, val_loss_grid_tc, val_mean_pck_tc, best_epoch_tc, best_val_pck_tc)
        msg_merge = 'Merge: Until Epoch {}\tBest Val pck {:.6f}\tBest epoch {}\tBest model type {}\tcost time {:.1f} min\n'.format(epoch,best_val_pck_merge, best_epoch_merge,best_model_type, (time.time() - trn_ep_st)/60)
        msg_all = msg_stu + msg_tc + msg_merge
        Logger.info_np(msg_all+'\n')
        Logger.info_main(msg_all+'\n')
        test_writer.add_text('summary', msg_all, epoch)


    Logger.info_np('seed {}, Training took: {} mins'.format(args.seed, (time.time()-train_started)//60))
    Logger.info_main('seed {}, Training took: {} mins'.format(args.seed, (time.time()-train_started)//60))
