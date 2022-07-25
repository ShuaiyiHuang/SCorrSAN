import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils_training.utils import flow2kps
from utils_training.evaluation import Evaluator

r'''
    EPE loss function implementation from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):
    EPE_map = torch.norm(target_flow-input_flow, 2, 1) # (b, 16, 16)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0) # (b, 16, 16)

        EPE_map = EPE_map[~mask] # (n, ) selected grids
    if mean: # mean is used
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(~mask)

# Label densification and filtering
def EPE_pseudo_filtered_by_dilation(input_flow_stu, input_flow_tc, target_flow, pseudo_ksize, avg=True):
    # exchange supervision
    target_flow_for_stu = input_flow_tc.detach().clone()
    target_flow_for_tc = input_flow_stu.detach().clone()

    EPE_map_stu = torch.norm(target_flow_for_stu - input_flow_stu, 2, 1)
    EPE_map_tc = torch.norm(target_flow_for_tc - input_flow_tc, 2, 1)

    batch_size = EPE_map_stu.size(0)

    # invalid flow is defined with both flow coordinates to be exactly 0
    mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

    mask_labeled = (1 - mask.float()).unsqueeze(1) # (b, 1, h, w)

    # dilate sparse label masks 
    strel_tensor = torch.ones((1, 1, pseudo_ksize, pseudo_ksize), dtype=torch.float) # (1,1,k,k)
    if EPE_map_stu.is_cuda:
        strel_tensor = strel_tensor.cuda(EPE_map_stu.get_device())

    mask_labeled_dilate = F.conv2d(mask_labeled,weight=strel_tensor,padding=(pseudo_ksize//2, pseudo_ksize//2)) # (b, 1, h, w)
    mask_labeled_dilate = mask_labeled_dilate > 0.0

    mask_labeled_dilate = mask_labeled_dilate.squeeze(1)

    EPE_map_stu = EPE_map_stu[mask_labeled_dilate] # (n, )
    EPE_map_tc = EPE_map_tc[mask_labeled_dilate] # (n, )
    
    if avg:
        loss_stu = EPE_map_stu.mean()
        loss_tc = EPE_map_tc.mean()
    else:
        loss_stu = EPE_map_stu
        loss_tc = EPE_map_tc

    return loss_stu, loss_tc

def train_epoch(net,
                optimizer,
                train_loader,
                device,
                epoch,
                train_writer, lmd_loss_pseudo, Logger,
                net_tc, optimizer_tc,remember_rate, pseudo_ksize, pstart_epoch):
    n_iter = epoch*len(train_loader)
    
    net.train()
    running_total_loss = 0

    # net_tc
    net_tc.train()
    running_total_loss_tc = 0

    # update lmd_loss_pseudo given pstart_epoch
    if not epoch>=pstart_epoch:
        lmd_loss_pseudo = 0.0
    Logger.info_main('Epoch {} start: lmd_loss_pseudo {}, rem_rate {}'.format(epoch, lmd_loss_pseudo,remember_rate))
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        optimizer_tc.zero_grad()

        flow_gt = mini_batch['flow'].to(device)

        pred_flow = net(mini_batch['trg_img'].to(device),
                         mini_batch['src_img'].to(device))
        
        pred_flow_tc = net_tc(mini_batch['trg_img'].to(device),
                         mini_batch['src_img'].to(device))

        # gt loss
        Loss_gt = EPE(pred_flow, flow_gt)
        Loss_gt_tc = EPE(pred_flow_tc, flow_gt)

        # pseudo-loss
        Loss_pseudo_raw, Loss_pseudo_tc_raw = EPE_pseudo_filtered_by_dilation(pred_flow, pred_flow_tc, flow_gt, pseudo_ksize, avg=False) # (n,)
        Loss_pseudo, Loss_pseudo_tc = loss_dynamic_selection(loss_1=Loss_pseudo_raw, loss_2=Loss_pseudo_tc_raw, remember_rate=remember_rate)
          
        Loss_tc = Loss_gt_tc + lmd_loss_pseudo*Loss_pseudo_tc 
        Loss = Loss_gt + lmd_loss_pseudo*Loss_pseudo 

        Loss.backward() 
        optimizer.step()

        Loss_tc.backward()
        optimizer_tc.step()

        running_total_loss += Loss.item()
        running_total_loss_tc +=Loss_tc.item()

        train_writer.add_scalar('train_loss_pseudo_per_iter', Loss_pseudo.item(), n_iter)
        
        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        train_writer.add_scalar('train_loss_per_iter_tc', Loss_tc.item(), n_iter)
        
        Logger.info_np('training: Epoch %d iter %d/%d: R_total_loss: %.3f/%.3f' % (epoch, i+1, len(train_loader), running_total_loss / (i + 1), Loss.item()))
        Logger.info_np('train_tc: Epoch %d iter %d/%d: R_total_loss: %.3f/%.3f' % (epoch, i+1, len(train_loader), running_total_loss_tc / (i + 1), Loss_tc.item()))

        n_iter += 1
        pbar.set_description('training: R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
    running_total_loss /= len(train_loader)
    running_total_loss_tc /=len(train_loader)
    return running_total_loss,running_total_loss_tc

def validate_epoch(net,
                   val_loader,
                   device,
                   epoch,
                   Logger = None,
                   mode = 'validation'):
    net.eval()
    running_total_loss = 0

    total_iters = len(val_loader)
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:

            flow_gt = mini_batch['flow'].to(device)
            pred_flow = net(mini_batch['trg_img'].to(device),
                            mini_batch['src_img'].to(device))

            estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)
            
            Loss = EPE(pred_flow, flow_gt) 

            pck_array += eval_result['pck']

            running_total_loss += Loss.item()

            if not Logger is None:
                Logger.info_np('%s: Epoch %d iter %d/%d: R_total_loss: %.3f/%.3f' % (mode, epoch, i+1, total_iters, running_total_loss / (i + 1), Loss.item()))
            pbar.set_description(' %s R_total_loss: %.3f/%.3f' % (mode, running_total_loss / (i + 1), Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)

    return running_total_loss / len(val_loader), mean_pck


r'''
    Select small-loss samples on pixel-level, modified from
    https://github.com/bhanML/Co-teaching/blob/master/loss.py
'''

def loss_dynamic_selection(loss_1, loss_2, remember_rate):

    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    num_remember = int(remember_rate * len(loss_1_sorted))
    if num_remember<1: 
        num_remember = 1

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # exchange
    loss_1_update = loss_1[ind_2_update]
    loss_2_update = loss_2[ind_1_update]

    # final loss
    loss_1_final = torch.sum(loss_1_update) / num_remember
    loss_2_final = torch.sum(loss_2_update) / num_remember
   
    return loss_1_final, loss_2_final